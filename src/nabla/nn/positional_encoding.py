from __future__ import annotations

import numpy as np

from nabla.nn.module import Module
from nabla.tensor import Tensor


def _sinusoidal_table(max_len: int, embed_dim: int) -> np.ndarray:
    """The fixed sin/cos position table from "Attention Is All You Need".

    table[pos, 2i]   = sin(pos / 10000^(2i / embed_dim))
    table[pos, 2i+1] = cos(pos / 10000^(2i / embed_dim))
    """
    position = np.arange(max_len)[:, None]  # (max_len, 1)
    i = np.arange(embed_dim)[None, :]  # (1, embed_dim)
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / embed_dim)
    angles = position * angle_rates  # (max_len, embed_dim)

    table = np.zeros((max_len, embed_dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return table


class PositionalEncoding(Module):
    """Adds a fixed sin/cos position signal to a sequence of embeddings.

    Attention itself has no notion of position - softmax(QK^T/sqrt(d_k))V
    would produce the exact same output if the sequence were shuffled and
    un-shuffled around it (every position attends to every other position
    symmetrically). Positional encoding is what breaks that symmetry: it's
    added directly to the token embeddings before they reach any attention
    layer, so position information is baked into the values Q/K/V get
    computed from.

    Unlike nn.Embedding, this table is NOT learned - it's a fixed
    mathematical pattern, precomputed once and just added on. That also
    means no new Function/backward is needed here: adding a constant only
    shifts the input, so Tensor's existing Add already provides the
    correct gradient (d(x+c)/dx = 1) automatically.

    Args:
        embed_dim: Size of the embedding vectors (must match the token
            embeddings this gets added to).
        max_len: Longest sequence length this table supports. The table is
            precomputed once for max_len positions and sliced down to
            however long the actual input sequence is.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        # a plain ndarray, not a Tensor: this is a fixed constant, never a
        # trainable parameter, so it must never show up in parameters()
        self.table = _sinusoidal_table(max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input with last dimension {self.embed_dim}, but got {x.data.shape[-1]}."
            )

        seq_len = x.data.shape[-2]
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        return x + Tensor(self.table[:seq_len])
