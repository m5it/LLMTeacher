#!/usr/bin/env python3
"""Start test training run with Option C: lr=1e-4, min_lr=5e-4, 50k iters"""
import json
from pathlib import Path

# Update config
config_path = Path("config/training_config.json")
config = json.loads(config_path.read_text())
config['learning_rate'] = 1e-4
config['min_lr'] = 5e-4
config['max_iters'] = 50000  # 50k for test
config_path.write_text(json.dumps(config, indent=2))
print(f"Updated config: lr={config['learning_rate']}, min_lr={config['min_lr']}, max_iters={config['max_iters']}")

# Import and run training
from llmteacher import train
print("\nStarting test training (50k iterations)...")
train(output_path="data/models/test_run_50k.pt")
