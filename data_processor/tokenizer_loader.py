import json
from pathlib import Path
from datetime import datetime

import tiktoken
from transformers import AutoTokenizer

_config_cache = None

def get_model_config():
    # Get the project root (where config/ dir is located)
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "model_config.json"
    return json.loads(config_path.read_text())

def get_training_config():
    return json.loads(Path("config/training_config.json").read_text())

def get_tokenizer():
    """Load tokenizer based on model_config.json 'tokenizer' field.
    Supported: gpt2, gemma3, llama, bert, bert-cased, t5, t5-large
    """
    config = get_model_config()
    name = config.get("tokenizer", "gpt2")

    if name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return enc, "gpt2", enc.n_vocab

    elif name == "gemma3":
        # Use LLaMA tokenizer (Gemma3 uses LLaMA-based tokenizer)
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "google/gemma-2b",  # Gemma3 uses same vocab as Gemma 2
            trust_remote_code=True,
            use_fast=True
        )
        return tokenizer, "gemma3", tokenizer.vocab_size

    elif name == "llama":
        # LLaMA 2 tokenizer (requires HuggingFace auth)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        return tokenizer, "llama", tokenizer.vocab_size

    elif name == "bert":
        # BERT uncased (110k vocab)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
        return tokenizer, "bert-uncased", tokenizer.vocab_size

    elif name == "bert-cased":
        # BERT cased (110k vocab)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", trust_remote_code=True)
        return tokenizer, "bert-cased", tokenizer.vocab_size

    elif name == "t5":
        # T5 small (32k vocab, SentencePiece)
        tokenizer = AutoTokenizer.from_pretrained("t5-small", trust_remote_code=True)
        return tokenizer, "t5-small", tokenizer.vocab_size

    elif name == "t5-large":
        # T5 large (32k vocab)
        tokenizer = AutoTokenizer.from_pretrained("t5-large", trust_remote_code=True)
        return tokenizer, "t5-large", tokenizer.vocab_size

    else:
        raise ValueError(f"Unknown tokenizer: {name}. Supported: gpt2, gemma3, llama, bert, bert-cased, t5, t5-large")

def get_tokenizer_name():
    """Return the tokenizer name from config."""
    config = get_model_config()
    return config.get("tokenizer", "gpt2")

def get_vocab_size():
    """Return vocabulary size for current tokenizer."""
    _, _, vocab_size = get_tokenizer()
    return vocab_size

def get_tokenizer_info():
    """Return dict with tokenizer name and vocab_size."""
    name = get_tokenizer_name()
    vocab_size = get_vocab_size()
    return {"tokenizer": name, "vocab_size": vocab_size}

def save_checkpoint_metadata(checkpoint_path, extra=None):
    """Save tokenizer and config info alongside checkpoint."""
    metadata = get_tokenizer_info()
    if extra:
        metadata.update(extra)
    meta_path = str(checkpoint_path).replace(".pt", "_meta.json")
    Path(meta_path).write_text(json.dumps(metadata, indent=2))
    return meta_path

# Tokenizer encode functions for ROCStories processing
def get_encode_fn(tokenizer_name):
    """Return appropriate encode function for each tokenizer type."""
    if tokenizer_name == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return lambda text: enc.encode_ordinary(text)

    elif tokenizer_name == "gemma3":
        tokenizer, _, _ = get_tokenizer()
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    elif tokenizer_name == "llama":
        tokenizer, _, _ = get_tokenizer()
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    elif tokenizer_name in ["bert", "bert-cased"]:
        tokenizer, _, _ = get_tokenizer()
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    elif tokenizer_name in ["t5", "t5-large"]:
        tokenizer, _, _ = get_tokenizer()
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

# Additional helper: Compare tokenization across tokenizers
def compare_tokenizers(text, tokenizers=["gpt2", "gemma3", "llama", "bert", "t5"]):
    """Compare how different tokenizers tokenize the same text."""
    results = {}
    for name in tokenizers:
        try:
            if name == "gpt2":
                enc = tiktoken.get_encoding("gpt2")
                tokens = enc.encode_ordinary(text)
                results[name] = {"tokens": tokens, "count": len(tokens), "ids": tokens[:10]}
            else:
                config = get_model_config()
                config["tokenizer"] = name
                tokenizer, _, vocab_size = get_tokenizer()
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                results[name] = {"tokens": token_ids, "count": len(token_ids), "vocab_size": vocab_size, "ids": token_ids[:10]}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# Log tokenizer usage to history
def log_tokenizer_usage(history_log="history.log"):
    """Append tokenizer info to history log."""
    info = get_tokenizer_info()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] TOKENIZER: {json.dumps(info)}\n"
    Path(history_log).open("a").write(entry)
