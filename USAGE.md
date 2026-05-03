# USAGE.md - LLMTeacher CLI#

## 🚀 Quick Start#
 
```bash
# 1. Choose tokenizer in config/model_config.json (default: gpt2)
# 2. Prepare data
python llmteacher.py prepare
 
# 3. Optional: Add ROCStories dataset
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
 
# 4. Preview configuration
python llmteacher.py preview
 
# 5. Train
python llmteacher.py train --output data/models/run1.pt
 
# 6. Generate text (auto-formats as instruction)
python llmteacher.py generate --latest --prompt "Once upon a time"
```
 
## 🆕 New Features#
 
### Download & Convert HuggingFace Models#
```bash
# Download pre-trained Gemma model from HuggingFace (requires HF auth + gated access)
python llmteacher.py download-model google/gemma-3-270m-it
 
# Convert HuggingFace model to LLMTeacher format
python llmteacher.py convert-model data/models/google_gemma-3-270m-it.pt
# Creates: data/models/google_gemma-3-270m-it_converted.pt
 
# Train on conversational data with converted model
python llmteacher.py train --checkpoint data/models/google_gemma-3-270m-it_converted.pt
```
 
### Fresh Training Start#
```bash
# Ignore previous training progress and start fresh (resets epoch counter)
python llmteacher.py train --checkpoint model.pt --fresh
python llmteacher.py continue model.pt --fresh
```
 
### Instruction Model Support#
- **Hardcoded instruction format**: All generation uses Gemma3-style chat format
  ```
  <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
  ```
- **Auto tokenizer detection**: Automatically uses Gemma tokenizer (256k vocab) or GPT-2 (50k vocab)
- **Conversational datasets**: Train on blended_skill_talk, daily_dialog, etc.
 
### Prepare Custom Data with Model Type#
```bash
# Prepare any text file as instruction or story format
python llmteacher.py prepare_random data/custom.txt --model-type instruction
python llmteacher.py prepare_random data/story.txt --model-type story
```

## 📋 Commands#
 
### `prepare`
Downloads TinyStories dataset and tokenizes to binary files using the tokenizer specified in `config/model_config.json`.
 
### `prepare-rocstories`
Processes ROCStories text files (`data/ROCStories/train*.txt`, `test.txt`) into binary token files. Uses the tokenizer from config.
 
### `combine-datasets`
Combines TinyStories and ROCStories into single `train_combined.bin` and `validation_combined.bin` in `data/processed_datasets/`.
 
### `preview`
Shows current configuration, available datasets with sample counts, and checkpoints.
```bash
python llmteacher.py preview
```
Displays: device status, model config, training config, tokenizer, datasets (with sizes and sample counts), checkpoints, and training progress.
 
### `datasets`
Lists all processed datasets with sample counts and file sizes.
```bash
python llmteacher.py datasets
```
Shows: dataset name, file size, description, and sample count (if metadata exists).
 
### `download-model`
Downloads pre-trained models from HuggingFace (requires HF auth + gated model access).
```bash
python llmteacher.py download-model google/gemma-3-270m-it
# Saves to: data/models/google_gemma-3-270m-it.pt
```
 
### `convert-model`
Converts HuggingFace model format to LLMTeacher format.
```bash
python llmteacher.py convert-model data/models/google_gemma-3-270m-it.pt
# Creates: data/models/google_gemma-3-270m-it_converted.pt
```
 
### `train`
Trains model from scratch. Options:
- `--checkpoint <path>` - Continue from a checkpoint (same as `continue`).
- `--output <path>` - Save checkpoint to a specific file. If not set, auto-generates a timestamped name like `data/models/model_lr{value}_{timestamp}.pt`.
- `--block-size <int>` - Override block size from config (e.g., 256 for conversations, 512 for stories).
- `--train-data <path>` - Use specific .bin file for training (overrides automatic dataset selection).
- `--device <string>` - Device to use (cuda, cpu, cuda:0).
- `--fresh` - Ignore training progress and start fresh (resets epoch counter).
 
### `continue`
Continues training from a checkpoint. First argument is the checkpoint path.
- `--output <path>` - Output checkpoint path (auto-generated if omitted).
- `--block-size <int>` - Override block size from config.
- `--train-data <path>` - Use specific .bin file for training.
- `--fresh` - Ignore training progress and start fresh.
 
Example:
```bash
python llmteacher.py continue data/models/run1.pt --output data/models/run2.pt
python llmteacher.py continue data/models/run1.pt --train-data data/processed_datasets/topical_chat_train.bin --block-size 256
```
 
### `generate`
Generates text from a prompt (auto-formatted as instruction). Options:
- `--prompt "text"` - Text prompt (default: "Once upon a time").
- `--checkpoint <path>` - Model checkpoint to use.
- `--latest` - Use the most recent checkpoint.
- `--model-type` - Model type (default: "instruction", hardcoded format).
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
Starts interactive chat with LLMTeacher (instruction format hardcoded). Options:
- `--checkpoint <path>` - Model checkpoint to use.
- `--latest` - Use the most recent checkpoint.
- `--model-type` - Model type (default: "instruction", hardcoded format).
- `--temperature <float>` - Sampling temperature (default: 0.8).
- `--top-k <int>` - Top-k sampling parameter (default: 50).
- `--max-tokens <int>` - Max tokens per response (default: 100).
 
Example:
```bash
python llmteacher.py chat --latest
You: Hello!
LLM: Hello there! How can I help you today?
```
 
**Note:** Generation always uses Gemma3 instruction format (`<start_of_turn>user...\n<start_of_turn>model\n`).
 
### `history`
Shows last N commands from `history.log`. Use `-n <int>` to change number of entries.
Tokenizer usage is automatically logged when training starts.
 
### `prepare_random <txt_path>`
Tokenizes any `.txt` file (one example per line) to `.bin` format for training.
- Creates `.bin` file and `.bin.meta.json` (for chunking support) in same directory as input.
- Uses tokenizer from config (gpt2/gemma3).
- `--model-type` - Format as "instruction" or "story" (default: "story").
 
Example:
```bash
python llmteacher.py prepare_random data/ROCStories/train1.txt
# Creates: data/ROCStories/train1.bin + train1.bin.meta.json
 
# Format as instruction data
python llmteacher.py prepare_random data/conversations.txt --model-type instruction
```
 
### `prepare-conv`
Processes conversational datasets (blended_skill_talk, discord, dailydialog) for chat training.
- `--dataset`: Choose dataset (blended_skill_talk, discord, dailydialog).
- `--max-samples`: Max conversations to process (default: 100,000).
- `--from-sample`, `--to-sample`: Slice dataset range (exclusive end).
 
Examples:
```bash
# Process blended_skill_talk from HuggingFace
python llmteacher.py prepare-conv --dataset blended_skill_talk --max-samples 5000
 
# Process Discord-Dialogues from HuggingFace
python llmteacher.py prepare-conv --dataset discord
 
# Slice dataset (process samples 10000-20000)
python llmteacher.py prepare-conv --dataset blended_skill_talk --from-sample 10000 --to-sample 20000
```
 
### `prepare-next-chunk`
Prepares next training chunk by skipping already-used stories (based on training progress).
- Reads `data/training_progress.json` to find last iteration.
- Uses metadata in `.bin.meta.json` to skip to new stories.
- Saves chunk to `data/processed_datasets/train_chunk.bin`.
- Training auto-uses chunk file if it exists.
 
Triggered automatically on Ctrl+C during training.
 
### `prepare-codesearch`
Downloads and processes CodeSearchNet dataset from HuggingFace.
- `--tokenizer`: Tokenizer to use (default: gpt2).
 
### `combine-with-code`
Combines existing training data with CodeSearchNet into `train_with_codesearchnet.bin`.

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

### Train on conversational dataset (Topical-Chat)#
```bash
# Prepare Topical-Chat dataset
python llmteacher.py prepare-conv --dataset topical_chat --max-samples 10000

# Train on conversation data only
python llmteacher.py train --train-data data/processed_datasets/topical_chat_train.bin --block-size 512

# Or combine with other datasets (default behavior)
python llmteacher.py train --block-size 512
```

### Use custom text file for training#
```bash
# Prepare any .txt file (one example per line)
python llmteacher.py prepare_random data/custom_stories.txt
# Creates: data/custom_stories.bin + custom_stories.bin.meta.json

# Train on custom data
python llmteacher.py train --train-data data/custom_stories.bin --block-size 512
```

### Dataset chunking (avoid repetition)#
```bash
# Start training
python llmteacher.py train --output data/models/run1.pt

# If interrupted (Ctrl+C), auto-generates new chunk
# Or manually prepare next chunk:
python llmteacher.py prepare-next-chunk

# Continue with fresh data
python llmteacher.py continue data/models/run1_interrupted.pt
```

### Preview configuration and datasets#
```bash
# Show full configuration
python llmteacher.py preview

# List all datasets with sample counts
python llmteacher.py datasets
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
    print(f"{name}: {data['count']} tokens, vocab={data.get('vocab_size', 'N/A')}")
"
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
│   ├── tokenizer_loader.py      # Multi-tokenizer support (6 tokenizers)
│   └── conversation_datasets.py # Conversational dataset processing
├── training/                   # Training utilities
│   ├── get_batch.py            # Batch generation (configurable via TRAIN_DATA_PATH)
│   ├── loss.py                 # Loss estimation
│   └── training_config.py      # Training config loader
├── data/                       # Data storage
│   ├── models/                 # Saved checkpoints + metadata
│   ├── processed_datasets/     # Processed binary data
│   ├── ROCStories/             # ROCStories text files
│   └── Topical-Chat/           # Topical-Chat conversational dataset
├── wandb/                      # Weights & Biases logs
└── results/                    # Training results + analysis
```
