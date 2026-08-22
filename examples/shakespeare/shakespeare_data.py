"""Downloads and tokenizes the Tiny Shakespeare corpus, cached locally.

Not part of the nabla package itself - this is dataset-specific glue for
the shakespeare_transformer.py example, kept separate from the library
like a real project would keep its dataset loading code next to the
training script that uses it. Tokenization here is deliberately the
simplest possible scheme (one integer per character) so the example needs
no separate tokenizer implementation - see docs/11-embedding.md for why
that's a different concern than the Embedding layer itself.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
from numpy.typing import NDArray

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(cache_dir: str) -> str:
    path = os.path.join(cache_dir, "input.txt")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Character-level tokenizer: vocabulary is just every distinct
    character that appears in the text, no merges or subword logic."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> NDArray:
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def decode(self, ids: NDArray) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_shakespeare(
    cache_dir: str = os.path.join(os.path.dirname(__file__), ".shakespeare_cache"),
    val_fraction: float = 0.1,
) -> tuple[NDArray, NDArray, CharTokenizer]:
    """Download (if needed) and tokenize Tiny Shakespeare.

    Returns:
        train_ids: Encoded training portion (first 1 - val_fraction of the text).
        val_ids: Encoded validation portion (the remainder).
        tokenizer: The CharTokenizer fit on the full text.
    """
    text = _download(cache_dir)
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)

    split = int(len(ids) * (1 - val_fraction))
    return ids[:split], ids[split:], tokenizer


def get_batch(
    ids: NDArray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    """Sample a random batch of (context, next-token-target) sequences.

    Returns:
        X: (batch_size, block_size) int64 - input token IDs.
        y: (batch_size, block_size) int64 - the same sequence shifted one
           position to the right, i.e. y[:, t] is the token that should
           follow X[:, t].
    """
    start_idx = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    X = np.stack([ids[i : i + block_size] for i in start_idx])
    y = np.stack([ids[i + 1 : i + block_size + 1] for i in start_idx])
    return X, y
