from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Embedding(Function):
    """Lookup table: out = table[indices].

    Shapes:
        table:   (vocab_size, embed_dim) - one learnable row per vocab
                 entry. This is the actual trainable parameter.
        indices: (*) any number of axes, integer values in
                 [0, vocab_size) - e.g. (batch, seq_len) for token IDs.
                 Not differentiable (like `targets` in
                 SoftmaxCrossEntropy) - integer IDs have no meaningful
                 gradient.
        out:     (*, embed_dim) - the corresponding row of `table` for
                 every index.
    """

    def forward(self, table: Tensor, indices: Tensor) -> NDArray:
        self.indices = indices.data
        self.table_shape = table.data.shape
        return table.data[indices.data]

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        # scatter-accumulate: a repeated index must sum its gradient
        # contributions, not have the last one overwrite the others -
        # same np.add.at pattern as MaxPool2D's backward
        xp = get_array_module(grad_output)
        grad_table = xp.zeros(self.table_shape)
        xp.add.at(grad_table, self.indices, grad_output)

        # indices are integer IDs, never meaningful to differentiate
        grad_indices = xp.zeros_like(self.indices, dtype=float)
        return grad_table, grad_indices
