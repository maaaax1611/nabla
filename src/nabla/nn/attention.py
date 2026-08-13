from __future__ import annotations

from typing import Callable

from numpy.typing import NDArray

from nabla.functional import scaled_dot_product_attention
from nabla.init import he
from nabla.nn.linear import Linear
from nabla.nn.module import Module
from nabla.tensor import Tensor


class Attention(Module):
    """Multi-head (scaled dot-product) attention, generalizing plain
    single-head attention as the num_heads=1 case - with learned Q/K/V/O
    projections, single-head attention isn't a special case in the code at
    all, just a different value of num_heads.

    Args:
        embed_dim: Size of the input/output feature dimension.
        num_heads: Number of parallel attention heads. embed_dim must be
            divisible by num_heads - each head operates on its own
            embed_dim / num_heads slice of the projected Q/K/V.
        weight_init: Passed through to the four internal Linear layers.

    Supports both self-attention (call with just `query`) and
    cross-attention (pass `key`/`value` explicitly, e.g. an encoder's
    output) - the same forward either way, since Q/K/V are always
    projected independently.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        weight_init: Callable = he,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, weight_init=weight_init)
        self.k_proj = Linear(embed_dim, embed_dim, weight_init=weight_init)
        self.v_proj = Linear(embed_dim, embed_dim, weight_init=weight_init)
        self.out_proj = Linear(embed_dim, embed_dim, weight_init=weight_init)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)."""
        batch, seq_len, _ = x.data.shape
        x = x.reshape((batch, seq_len, self.num_heads, self.head_dim))
        return x.transpose((0, 2, 1, 3))

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(batch, num_heads, seq, head_dim) -> (batch, seq, embed_dim)."""
        batch, num_heads, seq_len, head_dim = x.data.shape
        x = x.transpose((0, 2, 1, 3))
        return x.reshape((batch, seq_len, num_heads * head_dim))

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        mask: NDArray | None = None,
    ) -> Tensor:
        if not isinstance(query, Tensor):
            raise TypeError("query must be a Tensor.")
        key = query if key is None else key
        value = query if value is None else value

        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        attended = scaled_dot_product_attention(Q, K, V, mask=mask)
        return self.out_proj(self._merge_heads(attended))
