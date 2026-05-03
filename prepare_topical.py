#!/usr/bin/env python3
"""Prepare conversational dataset (blended_skill_talk) for training."""
import json
from pathlib import Path
from datasets import load_dataset

def prepare_conversational_data(output_dir="data/processed_datasets", max_samples=5000):
    """Prepare blended_skill_talk dataset in instruction format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading blended_skill_talk dataset...")
    dataset = load_dataset('blended_skill_talk', split='train')
    print(f"Loaded {len(dataset)} samples")

    conversations = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break

        # Use free_messages (more natural conversations)
        messages = sample['free_messages']
        if len(messages) >= 2:
            # Format as instruction pairs
            for j in range(0, len(messages)-1, 2):
                if j+1 < len(messages):
                    user_msg = messages[j]
                    assistant_msg = messages[j+1]

                    # Format in Gemma instruction format
                    formatted = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n{assistant_msg}<end_of_turn>\n"
                    conversations.append(formatted)

        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} samples, {len(conversations)} conversation pairs...")

    print(f"\nTotal conversation pairs: {len(conversations)}")

    # Save as text file
    output_path = output_dir / "blended_skill_talk_conversations.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(conv + "\n")

    print(f"Saved to {output_path}")
    return str(output_path)

if __name__ == "__main__":
    prepare_conversational_data(max_samples=5000)
