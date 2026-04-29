# AGENTS.md

## Project Context
From-scratch PyTorch implementation of Gemma3 270M, pre-trained on TinyStories.

## Key Quirks
- Uses GPT-2 tokenizer (50k vocab) instead of official Gemma3 256k vocab: 10x faster tokenization, reduces embedding params from ~170M to ~32M, sufficient for English-only TinyStories.
- Model supports 32k context length, but training uses 128-token block size.
- Training requires W&B login (`wandb login`) before running `02_pre_training.py`.

## Entrypoints
- **Unified CLI**: `python run.py <command>` (preferred, replaces individual scripts)
  - `python run.py prepare` - Download and tokenize TinyStories dataset
  - `python run.py train [--checkpoint path]` - Train from scratch or continue from checkpoint
  - `python run.py continue <checkpoint>` - Continue training from checkpoint
  - `python run.py generate [--prompt text] [--checkpoint path]` - Generate text
  - `python run.py list-checkpoints` - List available model checkpoints
- Legacy scripts (still functional):
  - `python 01_data_preparation.py` (downloads TinyStories, tokenizes to binary, requires `datasets` not in `requirements.txt`)
  - `python 02_pre_training.py` (W&B logging, requires `wandb` not in `requirements.txt`)
  - `03_inference.ipynb` (Jupyter notebook)

## Config Files
- `config/model_config.json`: Model hyperparams (layers, attention type, RoPE bases, dimensions)
- `config/training_config.json`: Training hyperparams (LR, batch size, iterations, gradient accumulation)

## Dependencies
`requirements.txt` lists `torch`, `tiktoken`, `numpy`, `tqdm`. Add `wandb` and `datasets` manually for full functionality.

## Structure Notes
- `architecture/`: Core model implementation (`gemma3.py`, `model_config.py`)
- `block/`: Transformer building blocks (attention, feedforward, RoPE, RMS norm)
- `data_processor/`: Tokenization utilities
- `training/`: Batch generation, loss, training config
- `data/`: Processed binary datasets (`processed_datasets/`), model checkpoints (`models/`)
