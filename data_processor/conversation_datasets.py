"""
Conversational dataset loader for LLMTeacher.
Downloads and processes conversational datasets for natural chat training.
"""
import json
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

def download_discord_dialogues(max_samples=None):
    """Download Discord-Dialogues from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading Discord-Dialogues from HuggingFace...")
        dataset = load_dataset("aaronmoo12/Discord-Dialogues", split="train")
        return dataset, "discord"
    except Exception as e:
        print(f"Error loading Discord-Dialogues: {e}")
        return None, None

def download_dailydialog():
    """Download DailyDialog from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading DailyDialog from HuggingFace...")
        dataset = load_dataset("ConvLab/DailyDialog", split="train")
        return dataset, "dailydialog"
    except Exception as e:
        print(f"Error loading DailyDialog: {e}")
        return None, None

def download_topical_chat(data_dir="data/Topical-Chat/conversations"):
    """Load Topical-Chat dataset from local files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Topical-Chat directory not found: {data_dir}")
        return None, None

    all_conversations = []
    for json_file in ["train.json", "valid_freq.json", "test_freq.json"]:
        file_path = data_dir / json_file
        if file_path.exists():
            print(f"Loading {json_file}...")
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Data is a dict with conversation IDs as keys
                for conv_id, conv in data.items():
                    if "content" in conv:
                        all_conversations.append(conv["content"])
                print(f"  Loaded {len(data)} conversations")

    print(f"Total conversations: {len(all_conversations):,}")
    return all_conversations, "topical_chat"

def format_conversation_discord(example):
    """Format Discord conversation to text."""
    # Discord format: 'text' field with <|im_start|> tags
    text = example.get("text", "")
    if not text:
        return ""
    # Convert <|im_start|>user/assistant markers to User:/Assistant:
    text = text.replace("<|im_start|>user\n", "User: ")
    text = text.replace("<|im_start|>assistant\n", "Assistant: ")
    text = text.replace("<|im_end|>", "\n")
    return text.strip()

def format_conversation_dailydialog(example):
    """Format DailyDialog conversation to text."""
    dialog = example.get("dialog", [])
    lines = []
    for i, utterance in enumerate(dialog):
        speaker = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{speaker}: {utterance}")
    return "\n".join(lines)

def format_conversation_topical_chat(conversation):
    """Format Topical-Chat conversation to text."""
    lines = []
    for turn in conversation:
        agent = "User" if turn.get("agent") == "agent_1" else "Assistant"
        message = turn.get("message", "")
        if message:
            lines.append(f"{agent}: {message}")
    return "\n".join(lines)

def process_conversational_data(tokenizer_name="gpt2", output_dir="data/processed_datasets",
                                 dataset_name="discord", max_samples=100000,
                                 from_sample=None, to_sample=None):
    """Process conversational dataset into binary format.

    Args:
        from_sample: Start index (inclusive) for slicing dataset
        to_sample: End index (exclusive) for slicing dataset
    """
    from data_processor.tokenizer_loader import get_encode_fn

    print(f"Processing {dataset_name} dataset with {tokenizer_name} tokenizer...")

    # Download/load dataset
    if dataset_name == "discord":
        dataset, name = download_discord_dialogues()
        format_fn = format_conversation_discord
    elif dataset_name == "dailydialog":
        dataset, name = download_dailydialog()
        format_fn = format_conversation_dailydialog
    elif dataset_name == "topical_chat":
        dataset, name = download_topical_chat()
        format_fn = format_conversation_topical_chat
    else:
        print(f"Unknown dataset: {dataset_name}")
        return False

    if dataset is None:
        return False

    encode_fn = get_encode_fn(tokenizer_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    story_boundaries = []
    current_pos = 0

    # Determine slice range
    if from_sample is not None or to_sample is not None:
        start = from_sample if from_sample is not None else 0
        end = to_sample if to_sample is not None else max_samples
        print(f"Processing samples [{start}:{end}]...")
        total = end - start
    else:
        start = 0
        end = max_samples
        print(f"Tokenizing conversations (max {max_samples:,} samples)...")
        total = max_samples

    count = 0
    for example in tqdm(dataset, desc="Processing conversations", total=total):
        # Skip samples before start
        if count < start:
            count += 1
            continue

        # Stop after end
        if count >= end:
            break

        text = format_fn(example)
        if text:
            tokens = encode_fn(text + "\n\n")
            all_tokens.extend(tokens)
            story_boundaries.append((current_pos, len(tokens)))
            current_pos += len(tokens)

        count += 1

    # Save as binary
    output_file = output_dir / f"{name}_train.bin"
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_file)
    print(f"Saved {len(all_tokens):,} tokens to {output_file} ({arr.nbytes / 1024 / 1024:.1f} MB)")

    # Save metadata for chunking
    meta = {
        "total_tokens": int(len(all_tokens)),
        "num_stories": len(story_boundaries),
        "story_boundaries": story_boundaries,
        "tokenizer": tokenizer_name
    }
    meta_path = output_file.with_suffix('.bin.meta.json')
    Path(meta_path).write_text(json.dumps(meta))
    print(f"Saved metadata to {meta_path}")

    return True

def combine_with_conversational_data():
    """Combine existing training data with conversational data."""
    output_dir = Path("data/processed_datasets")

    files = [
        output_dir / "train_combined.bin",  # TinyStories + ROCStories + CodeSearchNet
        output_dir / "discord_train.bin",
        output_dir / "dailydialog_train.bin",
        output_dir / "topical_chat_train.bin"
    ]

    print("Combining with conversational data...")
    train_data = []
    for f in files:
        if f.exists():
            data = np.fromfile(f, dtype=np.uint16)
            train_data.append(data)
            print(f"  Loaded {len(data):,} tokens from {f.name}")

    if train_data:
        combined = np.concatenate(train_data)
        output_path = output_dir / "train_with_conversation.bin"
        combined.tofile(output_path)
        print(f"Saved {len(combined):,} tokens to {output_path} ({combined.nbytes / 1024 / 1024:.1f} MB)")
        return True
    print("No conversational data found. Run process_conversational_data() first.")
    return False

if __name__ == "__main__":
    # Quick test
    print("Processing Discord-Dialogues (100k samples)...")
    if process_conversational_data(dataset_name="discord", max_samples=100000):
        combine_with_conversational_data()
