import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import torch
import numpy as np

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
#block size = context window

def get_batch(split, block_size, batch_size, device, device_type=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    # Check for combined dataset first, then fall back to original
    if split == 'train':
        combined_path = 'data/processed_datasets/train_combined.bin'
        original_path = 'data/processed_datasets/train.bin'
        data_path = combined_path if os.path.isfile(combined_path) else original_path
    else:
        combined_path = 'data/processed_datasets/validation_combined.bin'
        original_path = 'data/processed_datasets/validation.bin'
        data_path = combined_path if os.path.isfile(combined_path) else original_path
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to device
    if device_type == 'cuda' or 'cuda' in str(device):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
