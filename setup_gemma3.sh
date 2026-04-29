#!/bin/bash
cd /home/t3ch/models/llmteacher
echo "Starting Gemma3 tokenizer preparation..."
python llmteacher.py prepare
echo "Preparing ROCStories..."
python llmteacher.py prepare-rocstories
echo "Combining datasets..."
python llmteacher.py combine-datasets
echo "Done! Ready to train with Gemma3 tokenizer (256k vocab, ~296M params)"
