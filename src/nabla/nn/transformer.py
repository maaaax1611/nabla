from __future__ import annotations

from numpy.typing import NDArray

from nabla.nn.attention import Attention
from nabla.nn.dropout import Dropout
from nabla.nn.feedforward import FeedForward
from nabla.nn.layernorm import LayerNorm
from nabla.nn.module import Module
from nabla.tensor import Tensor


class TransformerBlock(Module):
    """One encoder-style Transformer block: self-attention + feedforward,
    each wrapped in a residual connection and pre-normalization.

    x = x + Dropout(Attention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))

    "Pre-norm" (normalize *before* the sublayer, not after) is the modern
    convention (GPT-2 onwards) rather than the original Transformer paper's
    post-norm - it keeps the residual path a clean, unnormalized sum of
    every previous layer's output, which trains more stably at depth.

    The residual connections (`x + ...`) matter independently of norm
    placement: without them, gradients would have to flow back through
    every attention/feedforward layer in sequence, the same
    vanishing-gradient problem deep networks always face. With them,
    gradients always have a direct, unimpeded path straight back to the
    input via the `+`, no matter how many blocks are stacked.

    Args:
        embed_dim: Model dimension (must match Attention/LayerNorm/FeedForward).
        num_heads: Number of attention heads, passed to `Attention`.
        hidden_dim: Width of the FeedForward's hidden layer.
        dropout: Dropout probability applied after attention and after
            the feedforward sublayer.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = Attention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: NDArray | None = None) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        attended = self.attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attended)

        fed_forward = self.ff(self.norm2(x))
        x = x + self.dropout(fed_forward)

        return x
