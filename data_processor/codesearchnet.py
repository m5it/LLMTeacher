"""
CodeSearchNet dataset loader for LLMTeacher.
Downloads and processes CodeSearchNet data for code generation training.
"""
import json
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

def download_codesearchnet():
    """Download CodeSearchNet dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading CodeSearchNet from HuggingFace...")
        # Load Python subset as example
        dataset = load_dataset("github/CodeSearchNet", "python", split="train", trust_remote_code=True)
        return dataset
    except Exception as e:
        print(f"Error loading CodeSearchNet: {e}")
        print("Please install: pip install datasets")
        return None

def process_codesearchnet(tokenizer_name="gpt2", output_dir="data/processed_datasets"):
    """Process CodeSearchNet into binary format using specified tokenizer."""
    from data_processor.tokenizer_loader import get_encode_fn
    
    print(f"Processing CodeSearchNet with {tokenizer_name} tokenizer...")
    
    # Load dataset
    dataset = download_codesearchnet()
    if dataset is None:
        return False
    
    encode_fn = get_encode_fn(tokenizer_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process code snippets (use 'code' field)
    all_tokens = []
    print("Tokenizing code snippets...")
    for example in tqdm(dataset, desc="Processing CodeSearchNet"):
        code = example.get("code", "")
        if code:
            tokens = encode_fn(code)
            all_tokens.extend(tokens)
    
    # Save as binary
    output_file = output_dir / "codesearchnet_train.bin"
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_file)
    print(f"Saved {len(all_tokens):,} tokens to {output_file} ({arr.nbytes / 1024 / 1024:.1f} MB)")
    return True

def combine_with_codesearchnet():
    """Combine existing training data with CodeSearchNet."""
    output_dir = Path("data/processed_datasets")
    
    files = [
        output_dir / "train_combined.bin",  # TinyStories + ROCStories
        output_dir / "codesearchnet_train.bin"
    ]
    
    print("Combining with CodeSearchNet...")
    train_data = []
    for f in files:
        if f.exists():
            data = np.fromfile(f, dtype=np.uint16)
            train_data.append(data)
            print(f"  Loaded {len(data):,} tokens from {f.name}")
    
    if train_data:
        combined = np.concatenate(train_data)
        output_path = output_dir / "train_with_code.bin"
        combined.tofile(output_path)
        print(f"Saved {len(combined):,} tokens to {output_path} ({combined.nbytes / 1024 / 1024:.1f} MB)")
        return True
    return False

if __name__ == "__main__":
    # Quick test
    if process_codesearchnet():
        combine_with_codesearchnet()
