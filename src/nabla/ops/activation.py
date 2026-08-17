from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class ReLU(Function):
    """ReLU activation"""
    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        xp = get_array_module(x.data)
        return xp.maximum(0, x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        return (grad_output * (x.data > 0),)


class Sigmoid(Function):
    """Sigmoid activation"""
    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        sigmoid = 1 / (1 + xp.exp(-x.data))
        self.save_for_backward(sigmoid)
        return sigmoid

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (sigmoid,) = self.saved_tensors
        return (grad_output * sigmoid * (1 - sigmoid),)
