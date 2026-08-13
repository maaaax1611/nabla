from __future__ import annotations

from nabla.nn.linear import Linear
from nabla.nn.module import Module
from nabla.tensor import Tensor


class FeedForward(Module):
    """Position-wise feedforward network: Linear -> ReLU -> Linear.

    "Position-wise" because it's applied independently to every position in
    the sequence (the same two Linear layers for every token, no mixing
    across the sequence axis - that's what attention already did). This is
    the standard Transformer FFN block: a projection up to a wider hidden
    dimension, a nonlinearity, and a projection back down.

    Args:
        embed_dim: Input/output feature dimension.
        hidden_dim: Width of the intermediate layer (conventionally
            4 * embed_dim in the original Transformer).
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())
