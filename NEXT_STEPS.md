# LLMTeacher - Next Steps After Current Test Run

## Current Status
- Test run (50k iters, Option C: lr=1e-4 -> 5e-4) is running
- Should finish in ~30 min
- Checkpoint will be: `data/models/test_run_50k.pt`

## After Test Run Completes

### 1. Check Results
```bash
cd /home/t3ch/models/llmteacher
tail -20 training_test.log
```

### 2. Generate Test Output
```bash
python3 llmteacher.py generate --latest --prompt "Write a Python function" --max-tokens 200
```

### 3. Prepare CodeSearchNet
```bash
# Install dependencies if needed
pip install datasets

# Run CodeSearchNet preparation
python3 llmteacher.py prepare-codesearch --tokenizer gpt2
```

### 4. Combine with CodeSearchNet
```bash
python3 llmteacher.py combine-with-code
# This creates: train_with_code.bin
```

### 5. Train on Code + Stories (Same Model)
```bash
# Use the checkpoint from test run, continue training with code data
python3 llmteacher.py continue data/models/test_run_50k.pt --output data/models/with_code.pt
```

**Note:** The model architecture stays the same (270M params), but now training on:
- TinyStories (~471M tokens)
- ROCStories (~4M tokens) 
- CodeSearchNet (~2M code snippets)

## CodeSearchNet Info
- Source: https://github.com/github/CodeSearchNet
- 2M (comment, code) pairs
- Languages: Python, JavaScript, Go, Java, PHP, Ruby
- Great for training code generation capabilities

## Tokenizer Warning System
✅ Added warnings for:
- `train` command - checks checkpoint vs current tokenizer
- `generate` command - warns if mismatch
- `chat` command - warns if mismatch

Each checkpoint saves `_meta.json` with tokenizer info!

## New Commands Added
- `prepare-codesearch` - Download and process CodeSearchNet
- `combine-with-code` - Add code data to training set
- `chat` - Interactive chat with LLMTeacher
