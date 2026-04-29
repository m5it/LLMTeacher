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
import argparse
import torch
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime

# Project imports
from architecture import Gemma3Model, model_config
from training import (
    learning_rate, max_iters, warmup_steps, min_lr, eval_iters,
    batch_size, block_size, gradient_accumulation_steps, device,
    device_type, dtype, ctx, get_batch, estimate_loss
)
from data_processor import processor_gpt2_tokenizer, get_tokenizer, get_tokenizer_name, save_checkpoint_metadata, get_vocab_size, log_tokenizer_usage
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import wandb


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
    """Download TinyStories and tokenize to binary files."""
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    if os.path.exists("data/processed_datasets/train.bin"):
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
        filename = f'data/processed_datasets/{split}.bin'
        dtype_np = np.uint16
        arr = np.memmap(filename, dtype=dtype_np, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
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


def train(checkpoint_path=None, output_path=None):
    """Train model from scratch or continue from checkpoint."""
    wandb.login()

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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif checkpoint_path:
        print(f"Checkpoint not found: {checkpoint_path}. Training from scratch.")

    torch.set_default_device(device)
    torch.manual_seed(42)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-9
    )

    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    training_config_log = {
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "block_size": block_size,
        "tokenizer": get_tokenizer_name(),
    }

    best_val_loss = float('inf')
    best_model_path = output_path
    Path("data/models").mkdir(parents=True, exist_ok=True)

    with wandb.init(project="pretraining-gemma3_270b", config=training_config_log) as run:
        model = model.to(device)
        run.watch(model, log_freq=100)

        for epoch in tqdm(range(max_iters)):
            if epoch % eval_iters == 0 and epoch != 0:
                losses = estimate_loss(model, eval_iters, ctx, block_size, batch_size, device, device_type)
                train_loss, val_loss = losses['train'], losses['val']
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                print(f"Learning rate: {current_lr:.5f}")
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "learning_rate": current_lr,
                    "best_val_loss": best_val_loss
                }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    save_checkpoint_metadata(best_model_path, {"val_loss": val_loss, "epoch": epoch})
                    wandb.log({"best_model_saved_at_epoch": epoch, "best_val_loss": best_val_loss}, step=epoch)

            X, y = get_batch("train", block_size, batch_size, device, device_type)
            X, y = X.to(device), y.to(device)

            with ctx:
                logits, loss = model(X, y)
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                wandb.log({"batch_loss": loss.item()}, step=epoch)

            if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                wandb.log({"grad_norm": grad_norm}, step=epoch)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    print("Training complete.")


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
    print(f"  Device: {device}")
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

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
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
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

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

    # Prepare CodeSearchNet
    codesearch_parser = subparsers.add_parser("prepare-codesearch", help="Download and process CodeSearchNet dataset")
    codesearch_parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use")

    # Combine with CodeSearchNet
    subparsers.add_parser("combine-with-code", help="Combine existing data with CodeSearchNet")

    # Train
    train_parser = subparsers.add_parser("train", help="Train model from scratch")
    train_parser.add_argument("--checkpoint", type=str, help="Continue training from checkpoint")
    train_parser.add_argument("--output", type=str, default=None, help="Output checkpoint path (auto-generated if not set)")

    # Continue (alias for train with checkpoint)
    continue_parser = subparsers.add_parser("continue", help="Continue training from checkpoint")
    continue_parser.add_argument("checkpoint", type=str, help="Checkpoint path to continue from")
    continue_parser.add_argument("--output", type=str, default=None, help="Output checkpoint path (auto-generated if not set)")

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
    elif args.command == "train":
        train(args.checkpoint, args.output)
    elif args.command == "continue":
        train(args.checkpoint, args.output)
    elif args.command == "generate":
        generate(args.prompt, args.checkpoint, args.latest, args.max_tokens, args.temperature, args.top_k)
    elif args.command == "list-checkpoints":
        list_checkpoints()
    elif args.command == "chat":
        chat(args.checkpoint, args.latest, args.temperature, args.top_k, args.max_tokens)
    elif args.command == "history":
        show_history(args.n)


def chat(checkpoint_path=None, use_latest=False, temperature=0.8, top_k=50, max_tokens=100):
    """Start interactive chat with LLMTeacher."""
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

    # Load model
    model_config["dtype"] = torch.bfloat16
    torch.manual_seed(123)
    model = Gemma3Model(model_config)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
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

        tokens = torch.tensor(history_tokens, dtype=torch.long, device=device).unsqueeze(0)

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
                if next_token.item() == (enc.eos_token_id if hasattr(enc, 'eos_token_id') else 0):
                    break

        response = decode_fn(generated_tokens)
        print(f"LLM: {response}\n")

        # Add response to history
        history_tokens.extend(generated_tokens)


if __name__ == "__main__":
    main()
