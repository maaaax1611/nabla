from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.broadcast import unbroadcast
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Add(Function):
    """Element-wise addition: z = x + y."""

    def forward(self, x: Tensor, y: Tensor) -> NDArray:
        self.save_for_backward(x, y)
        return x.data + y.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        x, y = self.saved_tensors
        grad_x = unbroadcast(grad_output, x.data.shape)
        grad_y = unbroadcast(grad_output, y.data.shape)
        return grad_x, grad_y


class Subtract(Function):
    """Element-wise subtraction: z = x - y."""

    def forward(self, x: Tensor, y: Tensor) -> NDArray:
        self.save_for_backward(x, y)
        return x.data - y.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        x, y = self.saved_tensors
        grad_x = unbroadcast(grad_output, x.data.shape)
        grad_y = unbroadcast(-grad_output, y.data.shape)
        return grad_x, grad_y


class Multiply(Function):
    """Element-wise multiplication: z = x * y."""

    def forward(self, x: Tensor, y: Tensor) -> NDArray:
        self.save_for_backward(x, y)
        return x.data * y.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        x, y = self.saved_tensors
        grad_x = unbroadcast(grad_output * y.data, x.data.shape)
        grad_y = unbroadcast(grad_output * x.data, y.data.shape)
        return grad_x, grad_y


class Divide(Function):
    """Element-wise division: z = x / y."""

    def forward(self, x: Tensor, y: Tensor) -> NDArray:
        self.save_for_backward(x, y)
        return x.data / y.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        x, y = self.saved_tensors
        grad_x = unbroadcast(grad_output / y.data, x.data.shape)
        grad_y = unbroadcast(-grad_output * x.data / (y.data ** 2), y.data.shape)
        return grad_x, grad_y