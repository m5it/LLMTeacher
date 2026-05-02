"""
Unified entrypoint for the Gemma3 270M TinyStories project.

Usage:
    python run.py prepare              # Download and tokenize TinyStories dataset
    python run.py prepare-rocstories  # Process ROCStories to binary format
    python run.py combine-datasets    # Combine TinyStories + ROCStories
    python run.py train                # Train from scratch
    python run.py continue --checkpoint data/models/best_model_params.pt  # Continue training
    python run.py generate --prompt "Once upon a time"  # Generate text (uses best_model_params.pt)
    python run.py generate --latest    # Generate text using most recent checkpoint
    python run.py generate --checkpoint data/models/model.pt  # Use specific checkpoint
    python run.py list-checkpoints     # List available model checkpoints
    python run.py history              # Show command history
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime

# GLOBAL DEVICE VARIABLE - set once and used everywhere
# ALWAYS default to CUDA - fail immediately if CUDA not available
# This ensures we NEVER accidentally use CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_TYPE = "cuda"
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("CUDA not available! This code requires CUDA. Check your PyTorch installation.")
CTX = None  # Will be set in train()/generate()/chat()

# Project imports
from architecture import Gemma3Model, model_config
from training import (
    learning_rate, max_iters, warmup_steps, min_lr, eval_iters,
    batch_size, block_size, gradient_accumulation_steps, dtype, get_batch, estimate_loss
)
from data_processor import processor_gpt2_tokenizer, get_tokenizer, get_tokenizer_name, save_checkpoint_metadata, get_vocab_size, log_tokenizer_usage
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from contextlib import nullcontext

# Optional W&B support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


HISTORY_LOG = Path("history.log")


def log_command(args):
    """Log command with timestamp to history.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = f"python run.py {args.command}"
    # Add all non-None arguments
    for arg, value in vars(args).items():
        if arg != "command" and value is not None and value is not False:
            if isinstance(value, bool) and value:
                cmd += f" --{arg}"
            elif not isinstance(value, bool):
                cmd += f" --{arg} {value}"
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {cmd}\n")


def show_history(n=10):
    """Show last n commands from history.log."""
    if not HISTORY_LOG.exists():
        print("No history found.")
        return
    lines = HISTORY_LOG.read_text().strip().split("\n")
    print(f"\nLast {min(n, len(lines))} commands:")
    print("-" * 60)
    for line in lines[-n:]:
        print(line)
    print("-" * 60)


def prepare_data():
    """Download TinyStories and tokenize to binary files with metadata for chunking."""
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    output_dir = Path("data/processed_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(output_dir / "train.bin"):
        print("Dataset already prepared. Skipping.")
        return

    tokenizer_name = get_tokenizer_name()
    print(f"Tokenizing dataset with {tokenizer_name} tokenizer...")

    if tokenizer_name == "gpt2":
        tokenized = ds.map(
            processor_gpt2_tokenizer,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )
    elif tokenizer_name == "gemma3":
        tokenized = ds.map(
            processor_gemma3_tokenizer,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = output_dir / f'{split}.bin'
        dtype_np = np.uint16
        arr = np.memmap(filename, dtype=dtype_np, mode='w+', shape=(arr_len,))

        # Store metadata: cumulative token positions for each story (for chunking)
        story_boundaries = []  # List of (story_index, start_token_position)
        current_pos = 0

        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch

            # Track story boundaries from this batch
            for i, story_len in enumerate(batch['len']):
                story_boundaries.append((idx + current_pos, story_len))
                current_pos += story_len

            idx += len(arr_batch)
        arr.flush()

        # Save metadata for chunking
        meta = {
            "total_tokens": int(arr_len),
            "num_stories": len(story_boundaries),
            "story_boundaries": story_boundaries,  # [(start_pos, length), ...]
            "tokenizer": tokenizer_name
        }
        meta_path = filename.with_suffix('.bin.meta.json')
        Path(meta_path).write_text(json.dumps(meta))
        print(f"Saved metadata to {meta_path}")

    print("Data preparation complete.")


def prepare_rocstories():
    """Process ROCStories text files to binary format."""
    tokenizer_name = get_tokenizer_name()
    print(f"Processing ROCStories with {tokenizer_name} tokenizer...")

    if tokenizer_name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda text: enc.encode_ordinary(text)
    elif tokenizer_name == "gemma3":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        encode_fn = lambda text: tokenizer.encode(text, add_special_tokens=False)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    roc_dir = Path("data/ROCStories")
    output_dir = Path("data/processed_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all train files (train.txt, train1.txt, etc.)
    train_files = sorted(roc_dir.glob("train*.txt"))
    if train_files:
        print(f"Processing ROCStories train ({len(train_files)} files)...")
        all_stories = []
        for fpath in train_files:
            with open(fpath, 'r') as f:
                stories = [line.strip() for line in f if line.strip()]
            all_stories.extend(stories)
            print(f"  Loaded {len(stories):,} stories from {fpath.name}")

        all_tokens = []
        for story in tqdm(all_stories, desc="Tokenizing train"):
            tokens = encode_fn(story)
            all_tokens.extend(tokens)

        output_file = output_dir / "rocstories_train.bin"
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_file)
        print(f"  Saved {len(all_tokens):,} tokens to {output_file} ({arr.nbytes / 1024 / 1024:.1f} MB)")
    else:
        print("No ROCStories train files found.")

    # Process test file
    test_file = roc_dir / "test.txt"
    if test_file.exists():
        print("Processing ROCStories test...")
        with open(test_file, 'r') as f:
            stories = [line.strip() for line in f if line.strip()]

        all_tokens = []
        for story in tqdm(stories, desc="Tokenizing test"):
            tokens = encode_fn(story)
            all_tokens.extend(tokens)

        output_file = output_dir / "rocstories_test.bin"
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(output_file)
        print(f"  Saved {len(all_tokens):,} tokens to {output_file} ({arr.nbytes / 1024 / 1024:.1f} MB)")
    else:
        print(f"File not found: {test_file}")

    print("ROCStories processing complete.")


def combine_datasets():
    """Combine TinyStories and ROCStories into single train/validation bins."""
    output_dir = Path("data/processed_datasets")

    # Combine train files
    train_files = [
        output_dir / "train.bin",
        output_dir / "rocstories_train.bin"
    ]

    print("Combining train datasets...")
    train_data = []
    for f in train_files:
        if f.exists():
            data = np.fromfile(f, dtype=np.uint16)
            train_data.append(data)
            print(f"  Loaded {len(data):,} tokens from {f.name}")

    if train_data:
        combined = np.concatenate(train_data)
        output_path = output_dir / "train_combined.bin"
        combined.tofile(output_path)
        print(f"  Saved {len(combined):,} tokens to {output_path} ({combined.nbytes / 1024 / 1024:.1f} MB)")

    # Combine validation files
    val_files = [
        output_dir / "validation.bin",
        output_dir / "rocstories_test.bin"
    ]

    print("Combining validation datasets...")
    val_data = []
    for f in val_files:
        if f.exists():
            data = np.fromfile(f, dtype=np.uint16)
            val_data.append(data)
            print(f"  Loaded {len(data):,} tokens from {f.name}")

    if val_data:
        combined = np.concatenate(val_data)
        output_path = output_dir / "validation_combined.bin"
        combined.tofile(output_path)
        print(f"  Saved {len(combined):,} tokens to {output_path} ({combined.nbytes / 1024 / 1024:.1f} MB)")

    print("Dataset combination complete.")

def prepare_random(txt_path):
    """Tokenize a .txt file to .bin format for training."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        print(f"File not found: {txt_path}")
        return

    bin_path = txt_path.with_suffix('.bin')
    meta_path = bin_path.with_suffix('').with_suffix('.bin.meta.json')

    print(f"Processing {txt_path.name}...")

    # Read all lines (one example per line)
    with open(txt_path, 'r') as f:
        examples = [line.strip() for line in f if line.strip()]

    print(f"Found {len(examples):,} examples")

    # Get tokenizer
    tokenizer_name = get_tokenizer_name()
    if tokenizer_name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda text: enc.encode_ordinary(text)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    # Tokenize all examples
    all_tokens = []
    story_boundaries = []
    current_pos = 0

    for example in tqdm(examples, desc="Tokenizing"):
        tokens = encode_fn(example)
        all_tokens.extend(tokens)
        story_boundaries.append((current_pos, len(tokens)))
        current_pos += len(tokens)

    # Save binary
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"Saved {len(all_tokens):,} tokens to {bin_path} ({arr.nbytes / 1024 / 1024:.1f} MB)")

    # Save metadata for chunking
    meta = {
        "total_tokens": int(len(all_tokens)),
        "num_stories": len(examples),
        "story_boundaries": story_boundaries,
        "tokenizer": tokenizer_name
    }
    meta_path.write_text(json.dumps(meta))
    print(f"Saved metadata to {meta_path}")


def prepare_next_chunk():
    """Prepare next data chunk using metadata to skip already-used stories."""
    from pathlib import Path
    progress_file = Path("data/training_progress.json")

    if not progress_file.exists():
        print("No progress file found. Run training first.")
        return

    try:
        progress = json.loads(progress_file.read_text())
        last_iter = progress.get("epoch", 0)  # epoch = iterations in our code
        print(f"Last training iteration: {last_iter:,}")
    except:
        print("Could not read progress file.")
        return

    # Load TinyStories metadata
    meta_path = Path("data/processed_datasets/train.bin.meta.json")
    if not meta_path.exists():
        print("No metadata found. Run prepare first.")
        return

    meta = json.loads(meta_path.read_text())
    story_boundaries = meta["story_boundaries"]

    # Calculate tokens processed: iterations * batch_size * block_size
    # Use values from config (or approximate)
    approx_tokens_processed = last_iter * 2 * 512  # batch_size=2, block_size=512
    print(f"Approx tokens processed: {approx_tokens_processed:,}")

    # Find which story this corresponds to
    stories_to_skip = 0
    for i, (pos, length) in enumerate(story_boundaries):
        if pos < approx_tokens_processed:
            stories_to_skip = i + 1
        else:
            break

    stories_to_skip = min(stories_to_skip, len(story_boundaries) - 1000)
    stories_to_skip = max(stories_to_skip, 0)

    print(f"Dataset has {len(story_boundaries):,} stories")
    print(f"Skipping first {stories_to_skip:,} stories for next chunk...")

    # Read original binary and skip to new starting point
    train_bin = Path("data/processed_datasets/train.bin")
    data = np.memmap(train_bin, dtype=np.uint16, mode='r')

    start_pos = story_boundaries[stories_to_skip][0] if stories_to_skip > 0 else 0
    remaining_tokens = len(data) - start_pos

    print(f"New chunk: {remaining_tokens:,} tokens (starting from story {stories_to_skip:,})")

    # Save new chunk
    chunk_path = Path("data/processed_datasets/train_chunk.bin")
    chunk_data = np.array(data[start_pos:], dtype=np.uint16)
    chunk_data.tofile(chunk_path)
    print(f"Saved chunk to {chunk_path} ({len(chunk_data):,} tokens)")

    print("\nNext chunk ready! Training will use the new chunk automatically.")
    print(f"Run: python llmteacher.py continue <checkpoint> --device cuda")

def train(checkpoint_path=None, output_path=None, device_arg=None, block_size_override=None, train_data_path=None):
    """Train model from scratch or continue from checkpoint."""
    global DEVICE, DEVICE_TYPE, CTX, block_size

    # Set custom train data path (for get_batch.py)
    if train_data_path:
        import training.get_batch as get_batch_module
        get_batch_module.TRAIN_DATA_PATH = train_data_path
        print(f"Training on custom data: {train_data_path}")

    # Use block_size from config, unless overridden by command line
    if block_size_override is not None:
        block_size = block_size_override
        print(f"Block size overridden to: {block_size}")
    else:
        print(f"Using block_size from config: {block_size}")
    
    # W&B login if available
    if WANDB_AVAILABLE:
        try:
            wandb.login()
        except:
            print("W&B login failed - continuing without W&B logging")
    
    # Show available devices
    print(f"\nAvailable devices:")
    print(f"  CPU: {os.cpu_count()} cores")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    else:
        print(f"  CUDA: Not available")
    
    # Device selection - only override global DEVICE if explicitly specified
    if device_arg:
        DEVICE = device_arg
        print(f"\nUsing specified device: {DEVICE}")
    else:
        print(f"\nUsing global DEVICE: {DEVICE}")
    
    # Update DEVICE_TYPE and CTX based on current DEVICE
    DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'
    dtype = 'bfloat16' if DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    CTX = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)
    
    print(f"✅ DEVICE confirmed: {DEVICE} (type: {DEVICE_TYPE})")
    print(f"   CTX: {CTX}")

    # Auto-generate output path if not specified
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr_str = str(learning_rate).replace('.', '').replace('-', 'neg')
        output_path = f"data/models/model_lr{lr_str}_{timestamp}.pt"
        print(f"Auto-generated output path: {output_path}")

    # Sync vocab_size with tokenizer
    vocab_size = get_vocab_size()
    if model_config.get("vocab_size") != vocab_size:
        print(f"Updating model_config vocab_size: {model_config.get('vocab_size')} -> {vocab_size}")
        model_config["vocab_size"] = vocab_size

    model_config["dtype"] = torch.bfloat16
    torch.manual_seed(123)
    model = Gemma3Model(model_config)

    # Check tokenizer consistency between checkpoint and current config
    current_tokenizer = get_tokenizer_name()
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Try to load metadata to check tokenizer
        meta_path = str(checkpoint_path).replace(".pt", "_meta.json")
        if os.path.exists(meta_path):
            try:
                meta = json.loads(Path(meta_path).read_text())
                checkpoint_tokenizer = meta.get("tokenizer", "unknown")
                if checkpoint_tokenizer != current_tokenizer:
                    print("\n" + "!" * 80)
                    print(f"⚠️  WARNING: TOKENIZER MISMATCH!")
                    print("!" * 80)
                    print(f"  Checkpoint was trained with tokenizer: {checkpoint_tokenizer}")
                    print(f"  Current config uses tokenizer:    {current_tokenizer}")
                    print()
                    print("  This will likely cause INCORRECT tokenization and BAD results!")
                    print()
                    print("  📝 Tips:")
                    print(f"     1. To continue with {checkpoint_tokenizer}, edit config/model_config.json:")
                    print(f'        "tokenizer": "{checkpoint_tokenizer}"')
                    print(f"     2. To switch to {current_tokenizer}, re-prepare data:")
                    print(f"        python llmteacher.py prepare")
                    print(f"        python llmteacher.py prepare-rocstories")
                    print(f"        python llmteacher.py combine-datasets")
                    print(f"     3. Or start fresh with current tokenizer:")
                    print(f"        python llmteacher.py train --output data/models/new_run.pt")
                    print("!" * 80 + "\n")
                    response = input("  Continue anyway? (yes/no): ")
                    if response.lower() != "yes":
                        print("Aborting. Please fix tokenizer mismatch first.")
                        exit(1)
                else:
                    print(f"✅ Tokenizer match: {current_tokenizer}")
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")
        else:
            print(f"\n⚠️  Warning: No metadata found for {checkpoint_path}")
            print(f"   Cannot verify tokenizer consistency. Current: {current_tokenizer}")
            print(f"   If checkpoint was trained with a different tokenizer, results will be wrong!\n")

        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    elif checkpoint_path:
        print(f"Checkpoint not found: {checkpoint_path}. Training from scratch.")

    torch.set_default_device(DEVICE)
    torch.manual_seed(42)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-9
    )

    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    
    # Move model to device FIRST
    model = model.to(DEVICE)
    
    # Suppress PyTorch warning about scheduler.step() before optimizer.step()
    # by doing a dummy optimizer step with actual model parameters
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    training_config_log = {
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "block_size": block_size,
        "tokenizer": get_tokenizer_name(),
        "device": DEVICE,
    }

    best_val_loss = float('inf')
    best_model_path = output_path
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # W&B init - skip if SSL errors
    run = None
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(project="pretraining-gemma3_270b", config=training_config_log)
            print("W&B logging enabled")
        except Exception as e:
            print(f"W&B init failed (continuing without W&B): {e}")
            run = None
    else:
        print("W&B not available, skipping...")
    
    # Model to device
    model = model.to(DEVICE)
    
    # Watch model if W&B available
    if run:
        run.watch(model, log_freq=100)
    
    # Confirmation prompt (only in interactive mode)
    if sys.stdin.isatty():
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Device: {DEVICE}")
        print(f"Checkpoint: {checkpoint_path or 'Training from scratch'}")
        print(f"Output: {output_path}")
        print(f"Learning rate: {learning_rate} (min_lr: {min_lr})")
        print(f"Iterations: {max_iters:,}")
        print(f"Batch size: {batch_size}, Block size: {block_size}")
        print(f"Tokenizer: {get_tokenizer_name()}")
        print(f"{'='*60}\n")
        response = input("Do you really want to continue? (Y/n): ")
        if response.lower() == 'n':
            print("Training cancelled.")
            return
    
    # Progress file path
    progress_file = Path("data/training_progress.json")
    
    # Load previous progress if continuing
    start_epoch = 0
    if checkpoint_path and os.path.exists(progress_file):
        try:
            progress = json.loads(progress_file.read_text())
            start_epoch = progress.get("epoch", 0)
            print(f"Resuming from epoch {start_epoch}")
        except:
            pass
    
    try:
        for epoch in tqdm(range(max_iters)):
            if epoch < start_epoch:
                continue  # Skip already processed epochs
            
            if epoch % eval_iters == 0 and epoch != 0:
                losses = estimate_loss(model, eval_iters, CTX, block_size, batch_size, DEVICE, DEVICE_TYPE)
                train_loss, val_loss = losses['train'], losses['val']
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                print(f"Learning rate: {current_lr:.5f}")
                if run:
                    wandb.log({
                        "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "learning_rate": current_lr,
                        "best_val_loss": best_val_loss
                    }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    save_checkpoint_metadata(best_model_path, {"val_loss": val_loss.item() if hasattr(val_loss, 'item') else val_loss, "epoch": epoch})
                    if run:
                        wandb.log({"best_model_saved_at_epoch": epoch, "best_val_loss": best_val_loss}, step=epoch)

            X, y = get_batch("train", block_size, batch_size, DEVICE, DEVICE_TYPE)
            X, y = X.to(DEVICE), y.to(DEVICE)

            with CTX:
                logits, loss = model(X, y)
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                if run:
                    wandb.log({"batch_loss": loss.item()}, step=epoch)

            if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if run:
                    wandb.log({"grad_norm": grad_norm}, step=epoch)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Now called AFTER optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                scheduler.step()  # Still call for iterations without optimizer step
            
            # Save progress periodically
            if epoch % 1000 == 0:
                progress_file.write_text(json.dumps({"epoch": epoch, "checkpoint": checkpoint_path or output_path}))
                
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"TRAINING INTERRUPTED AT EPOCH {epoch}")
        print(f"{'='*60}\n")

        # Save current state
        interrupt_path = output_path.replace(".pt", "_interrupted.pt")
        torch.save(model.state_dict(), interrupt_path)
        progress_file.write_text(json.dumps({"epoch": epoch, "checkpoint": output_path}))
        print(f"Saved interrupt checkpoint: {interrupt_path}")

        # Generate new dataset chunk from remaining data
        print(f"\nGenerating new dataset chunk from remaining stories...")
        prepare_next_chunk()

        print("\nTraining interrupted. To continue with new data, run:")
        print(f"  python llmteacher.py continue {interrupt_path} --device cuda\n")
        return
    
    # Save final progress
    progress_file.write_text(json.dumps({"epoch": max_iters, "checkpoint": output_path}))
    print("Training complete.")


def list_datasets():
    """List all processed datasets with sample counts and sizes."""
    from pathlib import Path
    import json

    print("\n" + "="*60)
    print(" "*15 + "Available Datasets")
    print("="*60)

    ds_dir = Path("data/processed_datasets")

    datasets = [
        ("train.bin", "TinyStories"),
        ("train_combined.bin", "TinyStories + ROCStories"),
        ("train_with_code.bin", "+ CodeSearchNet"),
        ("train_with_conversation.bin", "+ Conversational"),
        ("codesearchnet_train.bin", "CodeSearchNet"),
        ("discord_train.bin", "Discord-Dialogues"),
        ("dailydialog_train.bin", "DailyDialog"),
        ("topical_chat_train.bin", "Topical-Chat"),
        ("validation.bin", "Validation"),
        ("train_chunk.bin", "Chunk (remaining)"),
    ]

    for fname, desc in datasets:
        fpath = ds_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)

            # Try to load metadata for sample count
            meta_path = fpath.with_suffix('').with_suffix('.bin.meta.json')
            if not meta_path.exists():
                meta_path = fpath.parent / (fpath.stem + ".bin.meta.json")

            sample_info = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    samples = meta.get("num_stories", meta.get("num_samples", None))
                    if samples:
                        sample_info = f" [Samples: {samples:,}]"
                except:
                    pass

            print(f"  ✅ {fname:35s} ({size_mb:7.1f} MB) - {desc}{sample_info}")
        else:
            print(f"  ❌ {fname:35s} (not found) - {desc}")

    print("="*60)


def preview():
    """Show current configuration, datasets, and checkpoints."""
    from pathlib import Path
    import torch
    import json

    print("\n" + "="*60)
    print(" "*15 + "LLMTeacher Configuration Preview")
    print("="*60)

    # Device status
    print(f"\n🔧 DEVICE STATUS")
    if torch.cuda.is_available():
        print(f"  CUDA: ✅ {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  CUDA: ❌ Not available")
    print(f"  Config default: {DEVICE}")

    # Model config
    print(f"\n📦 MODEL CONFIG (config/model_config.json)")
    try:
        mc = json.loads(Path("config/model_config.json").read_text())
        print(f"  Vocab size: {mc.get('vocab_size', '?')} ({mc.get('tokenizer', '?')})")
        print(f"  Layers: {mc.get('n_layers', '?')} (sliding + full attention)")
        print(f"  Embedding dim: {mc.get('emb_dim', '?')}")
        print(f"  Heads: {mc.get('n_heads', '?')}, Head dim: {mc.get('head_dim', '?')}")
        print(f"  Context length: {mc.get('context_length', '?')}")
    except:
        print(f"  Config not found")

    # Training config
    print(f"\n⚙️  TRAINING CONFIG (config/training_config.json)")
    try:
        tc = json.loads(Path("config/training_config.json").read_text())
        print(f"  Learning rate: {tc.get('learning_rate', '?')} (min: {tc.get('min_lr', '?')})")
        print(f"  Max iterations: {tc.get('max_iters', '?')}")
        print(f"  Batch size: {tc.get('batch_size', '?')}, Block size: {tc.get('block_size', '?')}")
        print(f"  Gradient accumulation: {tc.get('gradient_accumulation_steps', '?')}")
        print(f"  Warmup steps: {tc.get('warmup_steps', '?')}")
        print(f"  Scheduler: {tc.get('lr_scheduler_type', '?')}")
    except:
        print(f"  Config not found")

    # Tokenizer
    print(f"\n🔤 TOKENIZER")
    print(f"  Type: {get_tokenizer_name()}")

    # Available datasets
    list_datasets()

    # Checkpoints
    print(f"\n💾 CHECKPOINTS (data/models/)")
    ckpt_dir = Path("data/models")
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob("*.pt"))
        if checkpoints:
            # Find latest
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"  Latest: {latest.name} ({latest.stat().st_size / 1024**2:.1f} MB)")
            print(f"  Total: {len(checkpoints)} checkpoints")
        else:
            print(f"  No checkpoints found")
    else:
        print(f"  No checkpoint directory")

    # Training progress
    progress_file = Path("data/training_progress.json")
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
            print(f"\n📈 TRAINING PROGRESS")
            print(f"  Last iteration: {progress.get('epoch', '?')}")
            print(f"  Checkpoint: {progress.get('checkpoint', '?')}")
        except:
            pass

    print("\n" + "="*60)
    print("Quick actions:")
    print("  Train:    python llmteacher.py train --block-size 512")
    print("  Generate: python llmteacher.py generate --latest --prompt \"Once upon\"")
    print("  Chat:     python llmteacher.py chat --latest")
    print("="*60 + "\n")


def get_latest_checkpoint():
    """Get the most recently modified checkpoint file."""
    checkpoint_dir = Path("data/models")
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def generate(prompt, checkpoint_path=None, use_latest=False, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt."""
    global DEVICE, DEVICE_TYPE, CTX
    # Ensure DEVICE is set
    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'
        print(f"Device auto-set to: {DEVICE}")
    
    if use_latest:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path:
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found.")
            return
    elif not checkpoint_path:
        checkpoint_path = "data/models/best_model_params.pt"

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please train first.")
        available = list_checkpoints_str()
        if available:
            print(f"Available checkpoints:\n{available}")
        return

    # Show model info
    print(f"\n{'='*60}")
    print(f"MODEL INFO")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    checkpoint_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    print(f"Checkpoint size: {checkpoint_size:.1f} MB")

    # Load checkpoint to check vocab size - use DEVICE not hardcoded cpu
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    ckpt_vocab_size = ckpt['tok_emb.weight'].shape[0]
    if ckpt_vocab_size != model_config['vocab_size']:
        print(f"Adjusting vocab_size from {model_config['vocab_size']} to {ckpt_vocab_size} (from checkpoint)")
        model_config['vocab_size'] = ckpt_vocab_size

    # Load model to get parameter count
    model_config["dtype"] = torch.bfloat16
    model = Gemma3Model(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Show model config
    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {model_config['vocab_size']:,}")
    print(f"  Embedding dimension: {model_config['emb_dim']}")
    print(f"  Layers: {model_config['n_layers']}")
    print(f"  Attention heads: {model_config['n_heads']}")
    print(f"  Head dimension: {model_config['head_dim']}")
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
    print(f"  Context length: {model_config['context_length']:,}")

    # Show generation params
    print(f"\nGeneration Parameters:")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Check tokenizer consistency for generation
    current_tokenizer = get_tokenizer_name()
    meta_path = str(checkpoint_path).replace(".pt", "_meta.json")
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text())
            checkpoint_tokenizer = meta.get("tokenizer", "unknown")
            if checkpoint_tokenizer != current_tokenizer:
                print("\n" + "!" * 80)
                print("⚠️  WARNING: TOKENIZER MISMATCH!")
                print("!" * 80)
                print(f"  Checkpoint was trained with tokenizer: {checkpoint_tokenizer}")
                print(f"  Current config uses tokenizer:      {current_tokenizer}")
                print("\n  This will generate INCORRECT text!")
                print("\n  💡 Tips:")
                print(f"     1. To use this checkpoint, edit config/model_config.json:")
                print(f'        "tokenizer": "{checkpoint_tokenizer}"')
                print(f"     2. Or load a checkpoint that matches your current tokenizer")
                print("!" * 80 + "\n")
        except:
            pass

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    tokenizer_name = get_tokenizer_name()
    if tokenizer_name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    elif tokenizer_name == "gemma3":
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) > block_size:
                tokens = tokens[:, -block_size:]
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

    generated = enc.decode(tokens[0].tolist())
    print(f"\nGenerated text:\n{generated}")
    return generated


def list_checkpoints():
    """List available model checkpoints with tokenizer info."""
    checkpoint_dir = Path("data/models")
    if not checkpoint_dir.exists():
        print("No checkpoints directory found.")
        return []

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found.")
        return []

    print("Available checkpoints:")
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        # Try to load metadata
        meta_path = ckpt.with_suffix('').with_suffix('.json')  # .pt -> .json
        meta_path = ckpt.parent / (ckpt.stem + "_meta.json")
        tokenizer_info = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                tokenizer_info = f" [tokenizer: {meta.get('tokenizer', '?')}]"
            except:
                pass
        print(f"  {ckpt.name} ({size_mb:.1f} MB){tokenizer_info}")
    return checkpoints


def list_checkpoints_str():
    """Return formatted string of available checkpoints."""
    checkpoint_dir = Path("data/models")
    if not checkpoint_dir.exists():
        return "No checkpoints directory found."

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return "No checkpoints found."

    lines = ["Available checkpoints:"]
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        lines.append(f"  {ckpt.name} ({size_mb:.1f} MB)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Gemma3 270M TinyStories CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare TinyStories
    subparsers.add_parser("prepare", help="Download and tokenize TinyStories dataset")

    # Prepare ROCStories
    subparsers.add_parser("prepare-rocstories", help="Process ROCStories to binary format")

    # Combine datasets
    subparsers.add_parser("combine-datasets", help="Combine TinyStories + ROCStories into single files")
    
    # Code dataset (separate - different block size!)
    codesearch_parser = subparsers.add_parser("prepare-codesearch", help="Download and process CodeSearchNet dataset")
    codesearch_parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train model from scratch")
    train_parser.add_argument("--checkpoint", type=str, help="Continue training from checkpoint")
    train_parser.add_argument("--output", type=str, default=None, help="Output checkpoint path (auto-generated if not set)")
    train_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, or cuda:0). Default: from config")
    train_parser.add_argument("--block-size", type=int, default=None,
                          help="Override block size from config (e.g. 128 for code, 512 for stories)")
    train_parser.add_argument("--train-data", type=str, default=None,
                          help="Path to specific .bin file for training (overrides automatic dataset selection)")

    # Continue (alias for train with checkpoint)
    continue_parser = subparsers.add_parser("continue", help="Continue training from checkpoint")
    continue_parser.add_argument("checkpoint", type=str, help="Checkpoint path to continue from")
    continue_parser.add_argument("--output", type=str, default=None, help="Output checkpoint path (auto-generated if not set)")
    continue_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu). Default: from config")
    continue_parser.add_argument("--block-size", type=int, default=None,
                          help="Override block size from config")
    continue_parser.add_argument("--train-data", type=str, default=None,
                          help="Path to specific .bin file for training")

    # Prepare random text file to .bin
    prepare_random_parser = subparsers.add_parser("prepare_random", help="Tokenize .txt file to .bin")
    prepare_random_parser.add_argument("txt_path", type=str, help="Path to .txt file (one example per line)")

    # Prepare conversational dataset (Discord, DailyDialog, Topical-Chat)
    conv_parser = subparsers.add_parser("prepare-conv", help="Process conversational dataset for chat training")
    conv_parser.add_argument("--dataset", type=str, default="discord",
                          choices=["discord", "dailydialog", "topical_chat"],
                          help="Conversational dataset to process")
    conv_parser.add_argument("--json-files", type=str, nargs="*",
                          help="Specific JSON files to load (relative to json-dir). If not set, uses defaults.")
    conv_parser.add_argument("--json-dir", type=str, default=None,
                          help="Directory containing JSON files (default: data/Topical-Chat/conversations for topical_chat)")
    conv_parser.add_argument("--from-sample", type=int, default=None,
                          help="Start sample index (for slicing dataset)")
    conv_parser.add_argument("--to-sample", type=int, default=None,
                          help="End sample index (exclusive, for slicing dataset)")
    conv_parser.add_argument("--max-samples", type=int, default=100000,
                          help="Maximum number of conversations to process (ignored if --from-sample/--to-sample set)")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    gen_parser.add_argument("--checkpoint", type=str, help="Model checkpoint to use")
    gen_parser.add_argument("--latest", action="store_true", help="Use the most recent checkpoint")
    gen_parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")

    # List checkpoints
    subparsers.add_parser("list-checkpoints", help="List available model checkpoints")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with LLMTeacher")
    chat_parser.add_argument("--checkpoint", type=str, help="Model checkpoint to use")
    chat_parser.add_argument("--latest", action="store_true", help="Use the most recent checkpoint")
    chat_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    chat_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per response")

    # History
    history_parser = subparsers.add_parser("history", help="Show command history")
    history_parser.add_argument("-n", type=int, default=10, help="Number of recent commands to show")

    # Preview configuration
    subparsers.add_parser("preview", help="Show current configuration, datasets, and checkpoints")

    # List datasets
    subparsers.add_parser("datasets", help="List all processed datasets with sample counts")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Log command (except history command to avoid recursion)
    if args.command != "history":
        log_command(args)

    if args.command == "prepare":
        prepare_data()
    elif args.command == "prepare-rocstories":
        prepare_rocstories()
    elif args.command == "combine-datasets":
        combine_datasets()
    elif args.command == "prepare-codesearch":
        from data_processor.codesearchnet import process_codesearchnet
        process_codesearchnet(tokenizer_name=args.tokenizer)
    elif args.command == "combine-with-code":
        from data_processor.codesearchnet import combine_with_codesearchnet
        combine_with_codesearchnet()
    elif args.command == "prepare-next-chunk":
        prepare_next_chunk()
    elif args.command == "prepare_random":
        prepare_random(args.txt_path)
    elif args.command == "datasets":
        list_datasets()
    elif args.command == "preview":
        preview()
    elif args.command == "train":
        train(args.checkpoint, args.output, args.device, args.block_size, args.train_data)
    elif args.command == "continue":
        train(args.checkpoint, args.output, args.device, args.block_size, args.train_data)
    elif args.command == "generate":
        generate(args.prompt, args.checkpoint, args.latest, args.max_tokens, args.temperature, args.top_k)
    elif args.command == "list-checkpoints":
        list_checkpoints()
    elif args.command == "chat":
        chat(args.checkpoint, args.latest, args.temperature, args.top_k, args.max_tokens)
    elif args.command == "prepare-conv":
        from data_processor.conversation_datasets import process_conversational_data, combine_with_conversational_data
        process_conversational_data(
            tokenizer_name="gpt2",
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            from_sample=args.from_sample,
            to_sample=args.to_sample
        )
        combine_with_conversational_data()
    elif args.command == "history":
        show_history(args.n)


def chat(checkpoint_path=None, use_latest=False, temperature=0.8, top_k=50, max_tokens=100):
    """Start interactive chat with LLMTeacher."""
    global DEVICE, DEVICE_TYPE, CTX
    # Ensure DEVICE is set
    if DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'
        print(f"Device auto-set to: {DEVICE}")
    
    if use_latest:
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path:
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found.")
            return
    elif not checkpoint_path:
        checkpoint_path = "data/models/best_model_params.pt"

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please train first.")
        available = list_checkpoints_str()
        if available:
            print(f"Available checkpoints:\n{available}")
        return

    # Check tokenizer consistency
    current_tokenizer = get_tokenizer_name()
    meta_path = str(checkpoint_path).replace(".pt", "_meta.json")
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text())
            checkpoint_tokenizer = meta.get("tokenizer", "unknown")
            if checkpoint_tokenizer != current_tokenizer:
                print("\n" + "!" * 80)
                print("⚠️  WARNING: TOKENIZER MISMATCH!")
                print("!" * 80)
                print(f"  Checkpoint was trained with tokenizer: {checkpoint_tokenizer}")
                print(f"  Current config uses tokenizer:    {current_tokenizer}")
                print("\n  This will generate INCORRECT text!")
                print("\n  💡 Tips:")
                print(f"     1. To use this checkpoint, edit config/model_config.json:")
                print(f'        "tokenizer": "{checkpoint_tokenizer}"')
                print(f"     2. Or load a checkpoint that matches your current tokenizer")
                print("!" * 80 + "\n")
        except:
            pass

    # Load model - check vocab size from checkpoint first
    model_config["dtype"] = torch.bfloat16
    
    # Auto-adjust vocab_size to match checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        ckpt_vocab_size = ckpt['tok_emb.weight'].shape[0]
        if ckpt_vocab_size != model_config.get('vocab_size', 0):
            print("\n" + "!" * 80)
            print("⚠️  VOCAB SIZE MISMATCH DETECTED!")
            print("!" * 80)
            print(f"  Checkpoint vocab_size: {ckpt_vocab_size:,}")
            print(f"  Current model_config vocab_size: {model_config.get('vocab_size')}")
            print(f"\n  This usually means the checkpoint was trained with a different tokenizer.")
            print(f"\n  To fix this, update config/model_config.json:")
            print(f"    1. Set vocab_size to {ckpt_vocab_size}")
            print(f"    2. Or set tokenizer to match the checkpoint")
            print(f"\n  Auto-adjusting vocab_size to {ckpt_vocab_size} for this session...")
            print("!" * 80 + "\n")
            model_config['vocab_size'] = ckpt_vocab_size
    
    torch.manual_seed(123)
    model = Gemma3Model(model_config)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Load tokenizer
    tokenizer_name = get_tokenizer_name()
    if tokenizer_name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        decode_fn = lambda tokens: enc.decode(tokens)
    elif tokenizer_name == "gemma3":
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        decode_fn = lambda tokens: enc.decode(tokens, skip_special_tokens=True)
    elif tokenizer_name in ["llama"]:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        decode_fn = lambda tokens: enc.decode(tokens, skip_special_tokens=True)
    elif tokenizer_name in ["bert", "bert-cased"]:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(f"bert-base-{tokenizer_name.replace('-', '')}", trust_remote_code=True)
        decode_fn = lambda tokens: enc.decode(tokens, skip_special_tokens=True)
    elif tokenizer_name in ["t5", "t5-large"]:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(f"{tokenizer_name}-small" if tokenizer_name == "t5" else "t5-large", trust_remote_code=True)
        decode_fn = lambda tokens: enc.decode(tokens, skip_special_tokens=True)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    print(f"\n{'='*60}")
    print(f"LLMTeacher Chat - Tokenizer: {tokenizer_name}")
    print(f"Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")

    # Chat loop
    history_tokens = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Tokenize and add to history
        if tokenizer_name == "gpt2":
            new_tokens = enc.encode_ordinary(user_input)
        else:
            new_tokens = enc.encode(user_input, add_special_tokens=False)

        history_tokens.extend(new_tokens)

        # Trim to block_size
        if len(history_tokens) > block_size:
            history_tokens = history_tokens[-block_size:]

        tokens = torch.tensor(history_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Generate response
        generated_tokens = []
        with torch.no_grad():
            for _ in range(max_tokens):
                if tokens.size(1) > block_size:
                    tokens = tokens[:, -block_size:]
                logits, _ = model(tokens)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat((tokens, next_token), dim=1)
                generated_tokens.append(next_token.item())
                eos_id = getattr(enc, 'eos_token_id', None)
                if tokenizer_name == "gpt2":
                    eos_id = 50256
                if eos_id is not None and next_token.item() == eos_id:
                    break

        response = decode_fn(generated_tokens)
        print(f"LLM: {response}\n")

        # Add response to history
        history_tokens.extend(generated_tokens)


if __name__ == "__main__":
    main()
