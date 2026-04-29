# USAGE.md - Gemma3 270M TinyStories CLI

## Quick Start

```bash
# 1. Download and tokenize TinyStories (GPT-2 tokenizer by default)
python llmteacher.py prepare

# 2. Process ROCStories (optional)
python llmteacher.py prepare-rocstories

# 3. Combine datasets (if using both)
python llmteacher.py combine-datasets

# 4. Train from scratch
python llmteacher.py train

# 5. Continue training from a checkpoint
python llmteacher.py continue data/models/best_model_params.pt

# 6. Generate text
python llmteacher.py generate --prompt "Once upon a time"
```

## Commands

### `prepare`
Downloads TinyStories dataset and tokenizes to binary files using the tokenizer specified in `config/model_config.json`.

### `prepare-rocstories`
Processes ROCStories text files (`data/ROCStories/train*.txt`, `test.txt`) into binary token files. Uses the tokenizer from config.

### `combine-datasets`
Combines TinyStories and ROCStories into single `train_combined.bin` and `validation_combined.bin` in `data/processed_datasets/`.

### `train`
Trains model from scratch. Options:
- `--checkpoint <path>` - Continue from a checkpoint (same as `continue`).
- `--output <path>` - Save checkpoint to a specific file. If not set, auto-generates a timestamped name like `data/models/model_lr{value}_{timestamp}.pt`.

### `continue`
Continues training from a checkpoint. First argument is the checkpoint path.
- `--output <path>` - Output checkpoint path (auto-generated if omitted).

Example:
```bash
python llmteacher.py continue data/models/best_model_params_lr5e-4_rocstories.pt --output data/models/model_round2.pt
```

### `generate`
Generates text from a prompt. Options:
- `--prompt "text"` - Text prompt (default: "Once upon a time").
- `--checkpoint <path>` - Model checkpoint to use.
- `--latest` - Use the most recent checkpoint.
- `--max-tokens <int>` - Max new tokens to generate (default: 100).
- `--temperature <float>` - Sampling temperature (default: 0.8).
- `--top-k <int>` - Top-k sampling parameter (default: 50).

Example:
```bash
python llmteacher.py generate --prompt "A little girl" --latest --max-tokens 200
```

### `list-checkpoints`
Lists all available checkpoints in `data/models/`. Shows file size and tokenizer info (if metadata exists).

### `history`
Shows last N commands from `history.log`. Use `-n <int>` to change number of entries.
Tokenizer usage is automatically logged when training starts.

## Configuration

### `config/model_config.json`
Main model hyperparameters. Important fields:
- `"tokenizer"`: `"gpt2"` (default) or `"gemma3"`. Determines which tokenizer to use.
- `"vocab_size"`: Automatically updated to match tokenizer when training starts.
- Other model architecture parameters.

### `config/training_config.json`
Training hyperparameters:
- `"learning_rate"`: Initial learning rate.
- `"min_lr"`: Minimum learning rate for cosine decay.
- `"max_iters"`: Total training iterations.
- `"batch_size"`, `"block_size"`, etc.
- `"tokenizer"`: Should match model_config (used for logging).

## Tokenizer Setup

The tokenizer is centralized in `config/model_config.json` with field `"tokenizer"`. Supported values:

| Tokenizer | Vocab Size | Model Used | Notes |
|-----------|------------|------------|-------|
| `"gpt2"` | 50,257 | GPT-2 | Fast, small embedding (~32M params) |
| `"gemma3"` | ~256k | Gemma 2B (LlamaTokenizerFast) | Matches Gemma3 vocab exactly |
| `"llama"` | 32,000 | Llama 2 7B | Requires HF auth `huggingface-cli login` |
| `"bert"` | 110,842 | BERT Base Uncased | Bidirectional encoder tokenizer |
| `"bert-cased"` | 110,842 | BERT Base Cased | Preserves case |
| `"t5"` | 32,128 | T5 Small (SentencePiece) | Prefix: `<extra_id_XX>` |
| `"t5-large"` | 32,128 | T5 Large (SentencePiece) | Larger model, same vocab |

**Important:** When you change tokenizer, re-run:
```bash
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
```

### Tokenizer Comparison
```bash
python -c "from data_processor.tokenizer_loader import compare_tokenizers; print(compare_tokenizers('Hello, world!'))"
```

## Checkpoint Management

- Each training run auto-saves the best model to the path specified by `--output` (or auto-generated).
- A metadata file `{checkpoint_name}_meta.json` is saved alongside the `.pt` file, containing tokenizer info and validation loss.
- `list-checkpoints` reads these metadata files and displays tokenizer info.
- To avoid overwriting, either use `--output` with a unique name or let auto-generation create a timestamped file.

## Background Training

For long runs, start training in background:

```bash
nohup python llmteacher.py continue data/models/checkpoint.pt > training.log 2>&1 &
```

Then monitor:
```bash
tail -f training.log
```

## Examples

### Full workflow with ROCStories
```bash
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
python llmteacher.py train --output data/models/first_run.pt
```

### Continue with different learning rate
Edit `config/training_config.json`, then:
```bash
python llmteacher.py continue data/models/first_run.pt --output data/models/second_run.pt
```

### Generate from latest checkpoint
```bash
python llmteacher.py generate --latest --prompt "The cat sat on"
```
