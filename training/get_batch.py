import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import torch
import numpy as np

# Module-level variable for configurable training data path
# Set this from llmteacher.py to use a specific .bin file for training
TRAIN_DATA_PATH = None

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
#block size = context window

def get_batch(split, block_size, batch_size, device, device_type=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    # Check for configurable train path first (set from llmteacher.py), then chunk, then combined, then original
    if split == 'train':
        if TRAIN_DATA_PATH and os.path.isfile(TRAIN_DATA_PATH):
            data_path = TRAIN_DATA_PATH
        else:
            chunk_path = 'data/processed_datasets/train_chunk.bin'
            if os.path.isfile(chunk_path):
                data_path = chunk_path
            else:
                combined_path = 'data/processed_datasets/train_with_codesearchnet.bin'
                if os.path.isfile(combined_path):
                    data_path = combined_path
                else:
                    combined_path2 = 'data/processed_datasets/train_combined.bin'
                    data_path = combined_path2 if os.path.isfile(combined_path2) else 'data/processed_datasets/train.bin'
    else:
        combined_path = 'data/processed_datasets/validation_combined.bin'
        data_path = combined_path if os.path.isfile(combined_path) else 'data/processed_datasets/validation.bin'
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    #
    #print("DEBUG get_batch.py device: {}".format(device))
    # Move to device
    if device_type == 'cuda' or 'cuda' in str(device):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
