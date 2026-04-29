# LLMTeacher - Gemma3 270M TinyStories

LLMTeacher is a from-scratch PyTorch implementation of Google DeepMind's Gemma3 270M model. It features a streamlined transformer architecture designed to help users understand and experiment with the core components of large language models.

**Objective:** This project aims to evaluate the coherence and language modeling capabilities of a small transformer-based model (Gemma3 270M) when trained from scratch on the TinyStories dataset. By leveraging TinyStories — a dataset of simple, coherent narratives — we can directly assess how well a compact language model learns to generate logical, contextually consistent, and fluent text.

## 🧠 Architecture

![architecture](./architecture/architecture.png)

Source: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3

## Overview

LLMTeacher implements the Gemma3 270M architecture — a 270-million parameter language model — pre-trained entirely from scratch on the TinyStories dataset.

### Parameter Counts by Tokenizer
| Tokenizer | Vocab Size | Embedding Params | Total Params |
|-----------|------------|------------------|--------------|
| gpt2 | 50,257 | ~32M | ~132M |
| gemma3 | ~256k | ~164M | ~296M |
| llama | 32,000 | ~20M | ~120M |
| bert / bert-cased | 110,842 | ~70M | ~170M |
| t5 / t5-large | 32,128 | ~20M | ~120M |

**Note:** The "270M" name comes from Gemma3's original 256k vocab. With smaller vocabs, total params decrease proportionally.

- **Mixed attention patterns**: Combination of sliding window and full attention layers  
- **Efficient architecture**: Optimized for low power consumption and on-device deployment
- **Strong instruction following**: Pre-trained with robust instruction-following capabilities
- **Large-scale training**: Official Gemma3 270M was trained on 6 trillion tokens

## 🔧 Tokenizer Support

LLMTeacher supports **6 tokenizers** via `config/model_config.json`:

| Tokenizer | Vocab Size | Model Used | Speed | Notes |
|-----------|------------|------------|-------|-------|
| `gpt2` | 50,257 | GPT-2 | ⚀ Fastest | Small embedding (~32M), great for English-only |
| `gemma3` | ~256k | Gemma 2B (LlamaTokenizerFast) | ⚀ Fast | Matches Gemma3 vocab exactly, requires `trust_remote_code=True` |
| `llama` | 32,000 | Llama 2 7B | ⚀ Fast | Requires HF auth: `huggingface-cli login` |
| `bert` | 110,842 | BERT Base Uncased | ⏱️ Medium | Bidirectional encoder, 110k vocab |
| `bert-cased` | 110,842 | BERT Base Cased | ⏱️ Medium | Preserves case |
| `t5` | 32,128 | T5 Small (SentencePiece) | ⏱️ Medium | Prefix: `<extra_id_XX>` |
| `t5-large` | 32,128 | T5 Large (SentencePiece) | ⏱️ Medium | Larger model, same vocab |

**Switching tokenizers:**
```bash
# 1. Edit config/model_config.json, change "tokenizer": "gpt2" to e.g. "gemma3"
# 2. Re-prepare data (this re-tokenizes everything)
python llmteacher.py prepare
python llmteacher.py prepare-rocstories
python llmteacher.py combine-datasets
# 3. Train with new vocab size
python llmteacher.py train --output data/models/gemma3_run.pt
```

## 🏗 Architecture Details

### Model Configuration
- **Vocabulary Size**: Configurable via tokenizer (50k-256k)
- **Context Length**: 32,768 tokens
- **Embedding Dimension**: 640
- **Number of Layers**: 18
- **Attention Heads**: 4
- **Hidden Dimension**: 2048
- **Head Dimension**: 256

### Key Features
1. **Grouped Query Attention (GQA)**: Efficient attention with configurable key-value groups
2. **RoPE (Rotary Position Embeddings)**: Dual RoPE bases for local (10,000) and global (1,000,000) context
3. **Sliding Window Attention**: 512-token sliding window for efficient local context
4. **Mixed Layer Types**: Strategic full attention layers (6, 12, 18) among sliding layers
5. **RMS Normalization**: Root Mean Square normalization with Gemma3-style scaling
6. **GELU Activation**: GELU activation in feedforward blocks

## 📂 Project Structure

```
llmteacher/
├── llmteacher.py              # Main CLI entrypoint
├── USAGE.md                  # Detailed usage guide
├── README.md                  # This file
├── AGENTS.md                  # Agent/developer context
├── architecture/               # Model architecture
│   ├── gemma3.py               # Main Gemma3Model class
│   └── model_config.py         # Model configuration loader
├── block/                      # Transformer blocks
│   ├── attention.py            # Grouped Query Attention
│   ├── feedforward.py          # ReLU feedforward
│   ├── transformer.py          # Transformer block
│   ├── rope.py                 # Rotary position embeddings
│   └── rms_norm.py             # RMS normalization
├── config/                     # Configuration
│   ├── model_config.json       # Model hyperparameters + tokenizer
│   └── training_config.json    # Training hyperparameters
├── data_processor/             # Data processing
│   ├── __init__.py
│   ├── process.py              # Tokenization functions
│   └── tokenizer_loader.py      # Multi-tokenizer support
├── training/                   # Training utilities
│   ├── get_batch.py            # Batch generation
│   ├── loss.py                 # Loss estimation
│   └── training_config.py      # Training config loader
├── data/                       # Data storage
│   ├── models/                 # Saved checkpoints + metadata
│   └── processed_datasets/     # Processed binary data
├── wandb/                      # Weights & Biases logs
└── results/                    # Training results + analysis
```

## 🚀 Quick Start

```bash
# 1. Choose tokenizer in config/model_config.json (default: gpt2)
# 2. Prepare data
python llmteacher.py prepare
python llmteacher.py prepare-rocstories  # optional

# 3. Combine datasets
python llmteacher.py combine-datasets

# 4. Train
python llmteacher.py train --output data/models/first_run.pt

# 5. Generate text
python llmteacher.py generate --latest --prompt "Once upon a time"
```

## 📊 Training Details

### Dataset
- **TinyStories**: ~471M tokens (GPT-2 tokenizer)
- **ROCStories**: ~4M tokens added (optional)
- **Total**: ~475M tokens combined
- Processed into memory-mapped binary files for efficient loading

### Training Configuration
- **Learning Rate**: 1e-4 with cosine annealing (configurable in `config/training_config.json`)
- **Warmup Steps**: 1,000
- **Max Iterations**: 150,000
- **Batch Size**: 2 (effective: 8 with gradient accumulation)
- **Block Size**: 128 tokens
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: bfloat16
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95, weight decay=0.1

### Training Features
- Weights & Biases integration for experiment tracking
- Gradient clipping (max norm 0.5)
- Best model checkpointing based on validation loss
- Mixed precision training with automatic scaling
- **Auto-generated checkpoint names** with timestamps
- **Metadata JSON** saved with each checkpoint (tokenizer, vocab_size, val_loss)

## 📊 Performance

The model achieves:
- Efficient training on consumer GPUs (16GB VRAM sufficient)
- Strong performance for its size class
- Low power consumption suitable for edge deployment
- Coherent story generation with proper grammar

### Training Results (GPT-2 tokenizer, 150k iterations)
- Final train loss: ~1.8 (perplexity ~6.0)
- Final validation loss: ~2.0 (perplexity ~7.4)
- Excellent generalization with no overfitting
- Coherent story generation with age-appropriate content

## 🛠️ Advanced Usage

### Compare Tokenizers
```bash
python -c "from data_processor.tokenizer_loader import compare_tokenizers; print(compare_tokenizers('Hello, world!'))"
```

### Check Available Checkpoints
```bash
python llmteacher.py list-checkpoints
# Shows: filename, size, tokenizer used
```

### Background Training
```bash
nohup python llmteacher.py continue data/models/checkpoint.pt > training.log 2>&1 &
tail -f training.log  # monitor progress
```

## 📦 Requirements
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended for larger vocabs)
- Python 3.12+
- Dependencies: datasets, tiktoken, transformers, wandb, numpy, tqdm

## 🔗 References
- [I pre-trained Gemma3 270M from scratch - tutorial from Vizuara](https://youtu.be/bLDlwcl6hbA)
- [Google DeepMind's Gemma3 270M model](https://huggingface.co/google/gemma-3-270m)
- [Official Gemma3 270M Blog Post](https://developers.googleblog.com/en/introducing-gemma-3-270m/)

## 💾 Tokenizer Comparison (Example)

| Tokenizer | "Hello, world!" tokens | Vocab Size |
|-----------|----------------------|------------|
| gpt2 | 4 tokens | 50,257 |
| gemma3 | 5 tokens | ~256,000 |
| llama | 6 tokens | 32,000 |
| bert-uncased | 8 tokens | 110,842 |
| t5 | 7 tokens | 32,128 |

Tokenizer choice affects both model size AND tokenization efficiency. Choose based on your needs!
