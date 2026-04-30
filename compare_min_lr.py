#!/usr/bin/env python3
"""Compare training with min_lr=5e-4 vs min_lr=1e-5"""
import json
import subprocess
import sys
from pathlib import Path

def run_training(min_lr, output_name, max_iters=3000):
    """Run a short training with given min_lr"""
    print(f"\n{'='*60}")
    print(f"Training with min_lr={min_lr}, iterations={max_iters}")
    print(f"{'='*60}\n")

    # Update config
    config_path = Path("config/training_config.json")
    config = json.loads(config_path.read_text())
    original_lr = config['learning_rate']
    original_min_lr = config['min_lr']
    original_iters = config['max_iters']

    config['learning_rate'] = 1e-4
    config['min_lr'] = min_lr
    config['max_iters'] = max_iters
    config_path.write_text(json.dumps(config, indent=2))

    # Run training
    from llmteacher import train
    checkpoint = f"data/models/compare_lr_{output_name}.pt"
    try:
        train(output_path=checkpoint)
    except Exception as e:
        print(f"Training failed: {e}")

    # Restore original config
    config['learning_rate'] = original_lr
    config['min_lr'] = original_min_lr
    config['max_iters'] = original_iters
    config_path.write_text(json.dumps(config, indent=2))

    return checkpoint

def generate_text(checkpoint_path, prompt="Once upon a time"):
    """Generate text from a checkpoint"""
    from llmteacher import generate
    return generate(prompt=prompt, checkpoint=checkpoint_path, max_tokens=100)

if __name__ == "__main__":
    # Run training with min_lr=5e-4
    ckpt_5e4 = run_training(min_lr=5e-4, output_name="minlr5e4", max_iters=3000)

    # Run training with min_lr=1e-5
    ckpt_1e5 = run_training(min_lr=1e-5, output_name="minlr1e5", max_iters=3000)

    # Compare generation
    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)

    print("\n--- min_lr=5e-4 ---")
    try:
        print(generate_text(ckpt_5e4))
    except Exception as e:
        print(f"Generation failed: {e}")

    print("\n--- min_lr=1e-5 ---")
    try:
        print(generate_text(ckpt_1e5))
    except Exception as e:
        print(f"Generation failed: {e}")
