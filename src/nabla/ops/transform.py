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

class Reshape(Function):
    """Reshape tensor to a new shape."""

    def __init__(self, new_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: Tensor) -> NDArray:
        self.original_shape = x.data.shape
        return x.data.reshape(self.new_shape)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        return (grad_output.reshape(self.original_shape),)


class Transpose(Function):
    """Transpose tensor (swap axes)."""

    def __init__(self, axes: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.axes = axes

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        return np.transpose(x.data, self.axes)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        if self.axes is None:
            # reversing all axes is its own inverse
            return (np.transpose(grad_output),)
        # recover the inverse permutation to undo the forward transpose
        inverse_axes = np.argsort(self.axes)
        return (np.transpose(grad_output, inverse_axes),)