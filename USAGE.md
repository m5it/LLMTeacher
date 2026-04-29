# USAGE.md - LLMTeacher CLI#

## 🚀 Quick Start#

```bash
# 1. Choose tokenizer in config/model_config.json (default: gpt2)
# 2. Prepare data
python llmteacher.py prepare

# 3. Optional: Add ROCStories dataset
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets

# 4. Train
python llmteacher.py train --output data/models/run1.pt

# 5. Generate text
python llmteacher.py generate --latest --prompt "Once upon a time"
```

## 📋 Commands#

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
python llmteacher.py continue data/models/run1.pt --output data/models/run2.pt
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

### `chat`
Starts interactive chat with LLMTeacher. Options:
- `--checkpoint <path>` - Model checkpoint to use.
- `--latest` - Use the most recent checkpoint.
- `--temperature <float>` - Sampling temperature (default: 0.8).
- `--top-k <int>` - Top-k sampling parameter (default: 50).
- `--max-tokens <int>` - Max tokens per response (default: 100).

Example:
```bash
python llmteacher.py chat --latest
You: Hello!
LLM: Hello there! How can I help you today?
```

**Warning:** If the checkpoint's tokenizer doesn't match current config, you'll get a warning and incorrect responses.

### `history`
Shows last N commands from `history.log`. Use `-n <int>` to change number of entries.
Tokenizer usage is automatically logged when training starts.

## ⚙️ Tokenizer Setup#

The tokenizer is centralized in `config/model_config.json` with field `"tokenizer"`. 

### Supported Tokenizers#
| Tokenizer | Vocab Size | Model Used | Speed | Embedding Params | Total Params | Notes |
|-----------|------------|------------|-------|------------------|--------------|-------|
| `"gpt2"` | 50,257 | GPT-2 | ⚀ Fastest | ~32M | ~132M | Great for English-only |
| `"gemma3"` | ~256k | Gemma 2B (LlamaTokenizerFast) | ⚀ Fast | ~164M | ~296M | Matches Gemma3 vocab exactly |
| `"llama"` | 32,000 | Llama 2 7B | ⚀ Fast | ~20M | ~120M | Requires HF auth |
| `"bert"` | 110,842 | BERT Base Uncased | ⏱️ Medium | ~70M | ~170M | Bidirectional encoder |
| `"bert-cased"` | 110,842 | BERT Base Cased | ⏱️ Medium | ~70M | ~170M | Preserves case |
| `"t5"` | 32,128 | T5 Small (SentencePiece) | ⏱️ Medium | ~20M | ~120M | Prefix: `<extra_id_XX>` |
| `"t5-large"` | 32,128 | T5 Large (SentencePiece) | ⏱️ Medium | ~20M | ~120M | Larger model, same vocab |

### Switching Tokenizers#
```bash
# 1. Edit config/model_config.json, change "tokenizer": "gpt2" to e.g. "gemma3"

# 2. Re-prepare data (this re-tokenizes everything)
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets

# 3. Train with new vocab size
python llmteacher.py train --output data/models/gemma3_run.pt
```

### Tokenizer Comparison#
Compare how different tokenizers tokenize the same text:
```bash
python -c "from data_processor.tokenizer_loader import compare_tokenizers; print(compare_tokenizers('Hello, world!'))"
```

## ⚙️ Configuration#

### `config/model_config.json`
Main model hyperparameters. Important fields:
- `"tokenizer"`: Tokenizer to use (gpt2, gemma3, llama, bert, bert-cased, t5, t5-large).
- `"vocab_size"`: Automatically updated to match tokenizer when training starts.
- Other model architecture parameters (emb_dim, n_layers, etc.).

### `config/training_config.json`
Training hyperparameters:
- `"learning_rate"`: Initial learning rate.
- `"min_lr"`: Minimum learning rate for cosine decay.
- `"max_iters"`: Total training iterations.
- `"batch_size"`, `"block_size"`, etc.
- `"tokenizer"`: Should match model_config (used for logging).

## 💾 Checkpoint Management#

- Each training run auto-saves the best model to the path specified by `--output` (or auto-generated).
- A metadata file `{checkpoint_name}_meta.json` is saved alongside the `.pt` file, containing:
  - Tokenizer name
  - Vocabulary size
  - Validation loss
  - Epoch number
- `list-checkpoints` reads these metadata files and displays tokenizer info.
- To avoid overwriting, either use `--output` with a unique name or let auto-generation create a timestamped file.

### Metadata Example#
```json
{
  "tokenizer": "gemma3",
  "vocab_size": 256000,
  "val_loss": 2.45,
  "epoch": 150000
}
```

## 🔧 Background Training#

For long runs, start training in background:

```bash
nohup python llmteacher.py continue data/models/checkpoint.pt > training.log 2>&1 &
```

Then monitor:
```bash
tail -f training.log
```

## 📝 Examples#

### Full workflow with ROCStories (GPT-2 tokenizer)#
```bash
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
python llmteacher.py train --output data/models/first_run.pt
```

### Switch to Gemma3 tokenizer and retrain#
```bash
# Edit config/model_config.json: "tokenizer": "gemma3"
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
python llmteacher.py train --output data/models/gemma3_run.pt
```

### Continue with different learning rate#
Edit `config/training_config.json`, then:
```bash
python llmteacher.py continue data/models/first_run.pt --output data/models/second_run.pt
```

### Generate from latest checkpoint#
```bash
python llmteacher.py generate --latest --prompt "The cat sat on"
```

### Compare all tokenizers#
```bash
python -c "
from data_processor.tokenizer_loader import compare_tokenizers
result = compare_tokenizers('Hello, world! This is LLMTeacher.')
for name, data in result.items():
    print(f\"{name}: {data['count']} tokens, vocab={data.get('vocab_size', 'N/A')}\")
"
```

## 📊 Project Structure#

```
llmteacher/
├── llmteacher.py              # Main CLI entrypoint
├── USAGE.md                  # This file
├── README.md                  # Project overview
├── AGENTS.md                  # Agent/developer context
├── setup_gemma3.sh            # Quick setup script for Gemma3 tokenizer
├── architecture/               # Model architecture
│   ├── gemma3.py               # Main Gemma3Model class
│   └── model_config.py         # Model configuration loader
├── block/                      # Transformer blocks
├── config/                     # Configuration
│   ├── model_config.json       # Model hyperparameters + tokenizer
│   └── training_config.json    # Training hyperparameters
├── data_processor/             # Data processing
│   ├── __init__.py
│   ├── process.py              # Tokenization functions
│   └── tokenizer_loader.py      # Multi-tokenizer support (6 tokenizers)
├── training/                   # Training utilities
│   ├── get_batch.py            # Batch generation
│   ├── loss.py                 # Loss estimation
│   └── training_config.py      # Training config loader
├── data/                       # Data storage
│   ├── models/                 # Saved checkpoints + metadata
│   ├── processed_datasets/     # Processed binary data
│   └── ROCStories/             # ROCStories text files
├── wandb/                      # Weights & Biases logs
└── results/                    # Training results + analysis
```
