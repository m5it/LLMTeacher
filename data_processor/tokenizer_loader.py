import json
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer

_config_cache = None

def get_model_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = json.loads(Path("config/model_config.json").read_text())
    return _config_cache

def get_training_config():
    return json.loads(Path("config/training_config.json").read_text())

def get_tokenizer():
    """Load tokenizer based on model_config.json 'tokenizer' field."""
    config = get_model_config()
    name = config.get("tokenizer", "gpt2")

    if name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return enc, "gpt2", enc.n_vocab
    elif name == "gemma3":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
        return tokenizer, "gemma3", tokenizer.vocab_size
    else:
        raise ValueError(f"Unknown tokenizer: {name}")

def get_tokenizer_name():
    """Return the tokenizer name from config."""
    config = get_model_config()
    return config.get("tokenizer", "gpt2")

def get_vocab_size():
    """Return vocabulary size for current tokenizer."""
    _, _, vocab_size = get_tokenizer()
    return vocab_size

def save_checkpoint_metadata(checkpoint_path, extra=None):
    """Save tokenizer and config info alongside checkpoint."""
    metadata = {
        "tokenizer": get_tokenizer_name(),
        "vocab_size": get_vocab_size(),
    }
    if extra:
        metadata.update(extra)
    meta_path = str(checkpoint_path).replace(".pt", "_meta.json")
    Path(meta_path).write_text(json.dumps(metadata, indent=2))
    return meta_path
