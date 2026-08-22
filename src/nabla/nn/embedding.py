from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from nabla.nn.module import Module
from nabla.ops.embedding import Embedding as EmbeddingFunction
from nabla.tensor import Tensor


def _small_normal(shape: tuple[int, ...]) -> NDArray:
    """Default embedding init: N(0, 0.01), like PyTorch's nn.Embedding.

    Unlike Linear/Conv2D, an embedding lookup isn't a matmul over all
    inputs - each output row depends on exactly one input row, so the
    fan_in/fan_out reasoning behind He/Xavier doesn't apply here.
    """
    return (np.random.randn(*shape) * 0.01).astype(np.float32)


class Embedding(Module):
    """Learnable lookup table mapping integer IDs to dense vectors.

    Args:
        vocab_size: Number of distinct IDs (rows in the table).
        embed_dim: Size of each embedding vector.
        weight_init: Callable mapping a shape to an initial weight array.
            Defaults to small-normal initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        weight_init: Callable[[tuple[int, ...]], NDArray] = _small_normal,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = Tensor(weight_init((vocab_size, embed_dim)), requires_grad=True)

    def forward(self, indices: Tensor) -> Tensor:
        if not isinstance(indices, Tensor):
            raise TypeError("indices must be a Tensor.")
        return EmbeddingFunction.apply(self.weight, indices)
