# Datasets

## Available Datasets

All processed datasets are stored in `data/processed_datasets/` in binary format (uint16 tokens).

| Dataset | File | Size | Tokens | Status |
|----------|------|------|--------|--------|
| TinyStories | `train.bin` | 900.0 MB | 471,872,517 | ✓ Prepared |
| TinyStories | `validation.bin` | 9.0 MB | 4,743,928 | ✓ Prepared |
| ROCStories | `rocstories_train.bin` | 7.7 MB | 4,032,675 | ✓ Prepared |
| ROCStories | `rocstories_test.bin` | 1.9 MB | 1,007,978 | ✓ Prepared |
| Combined | `train_combined.bin` | 907.7 MB | 475,905,192 | ✓ Prepared |
| Combined | `validation_combined.bin` | 11.0 MB | 5,751,906 | ✓ Prepared |
| CodeSearchNet | `codesearchnet_train.bin` | 379.2 MB | 198,817,643 | ✓ Prepared |

## Data Sources

- **TinyStories**: https://huggingface.co/datasets/roneneldan/TinyStories
  - Used for pre-training the model
  - English-only children's stories

- **ROCStories**: https://huggingface.co/datasets/instruct/rocstories
  - Used for continued training / fine-tuning
  - 5-sentence common-sense stories

- **CodeSearchNet**: https://huggingface.co/datasets/github/CodeSearchNet
  - Not yet prepared
  - Python code dataset for code generation training
  - To prepare: `python -c "from data_processor.codesearchnet import *; download_codesearchnet()"`

## Tokenizer

- **GPT-2 Tokenizer** (tiktoken)
  - Vocab size: 50,257
  - 10x faster tokenization vs Gemma3 tokenizer
  - Reduces embedding params from ~170M to ~32M

## Training Runs

| Run | Config | Checkpoint | Val Loss | Epoch | Status |
|-----|--------|------------|---------|-------|--------|
| Test 1 | lr=1e-4, min_lr=5e-4, 2000 iters | `model_lr00001_20260429_142846.pt` | 5.2479 | 500 | ✓ Complete |
| Test 2 | lr=1e-4, min_lr=1e-5, 2000 iters | `model_lr00001_20260429_143002.pt` | 4.9233 | 1500 | ✓ Complete |
| Full | lr=1e-4, min_lr=5e-4, 150k iters | `full_train_minlr5e4.pt` | 2.6813 | 144500 | ✓ Complete |

## CodeSearchNet Setup

Now working via HuggingFace (`code_search_net` dataset). To prepare:

```bash
# Process 500k samples (default, takes ~5 min)
python -c "from data_processor.codesearchnet import process_codesearchnet; process_codesearchnet()"

# Process more samples if needed
python -c "from data_processor.codesearchnet import process_codesearchnet; process_codesearchnet(max_samples=1000000)"
```

## Preparation Commands

```bash
# Prepare TinyStories dataset
python llmteacher.py prepare
# or
python 01_data_preparation.py

# Prepare ROCStories
python -c "from data_processor.rocstories import process_rocstories; process_rocstories()"

# Combine datasets
python -c "from data_processor.combine_datasets import combine_datasets; combine_datasets()"

# Prepare CodeSearchNet (optional)
python -c "from data_processor.codesearchnet import download_codesearchnet, process_codesearchnet; ds = download_codesearchnet(); process_codesearchnet()"
```
