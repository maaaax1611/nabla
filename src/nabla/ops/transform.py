from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class MatMul(Function):
    """Matrix multiplication: z = a @ b."""

    def forward(self, a: Tensor, b: Tensor) -> NDArray:
        self.save_for_backward(a, b)
        return np.dot(a.data, b.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        a, b = self.saved_tensors
        grad_a = np.dot(grad_output, b.data.T)
        grad_b = np.dot(a.data.T, grad_output)
        return grad_a, grad_b