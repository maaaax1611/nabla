from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Concat(Function):
    """Concatenate tensors along an existing axis: z = concat([a, b, ...], axis).

    Takes a variable number of tensors (not just 1 or 2) - Function.apply
    and Function.backward already support that via *inputs / tuple[...].
    """

    def __init__(self, axis: int = 0) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, *tensors: Tensor) -> NDArray:
        self.save_for_backward(*tensors)
        return np.concatenate([t.data for t in tensors], axis=self.axis)

    def backward(self, grad_output: NDArray) -> tuple[NDArray, ...]:
        # Each output element came from exactly one input (no broadcasting
        # overlap), so backward just cuts grad_output back into the same
        # pieces it was assembled from, in the same order.
        tensor_sizes = [t.data.shape[self.axis] for t in self.saved_tensors]
        split_indices = np.cumsum(tensor_sizes)[:-1]
        return tuple(np.split(grad_output, split_indices, axis=self.axis))
