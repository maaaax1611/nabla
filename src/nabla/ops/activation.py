from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class ReLU(Function):
    """ReLU activation"""
    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        return np.maximum(0, x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        return (grad_output * (x.data > 0),)


class Sigmoid(Function):
    """Sigmoid activation"""
    def forward(self, x: Tensor) -> NDArray:
        sigmoid = 1 / (1 + np.exp(-x.data))
        self.save_for_backward(sigmoid)
        return sigmoid

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (sigmoid,) = self.saved_tensors
        return (grad_output * sigmoid * (1 - sigmoid),)