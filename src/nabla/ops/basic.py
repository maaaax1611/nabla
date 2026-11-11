from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.backend import get_array_module
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


class Neg(Function):
    """Element-wise negation: z = -x."""

    def forward(self, x: Tensor) -> NDArray:
        return -x.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        return (-grad_output,)


class Pow(Function):
    """Element-wise power with a constant exponent: z = x ** p.

    The exponent is a plain Python number, not a Tensor - it's a fixed
    part of the op (like `axis` for Softmax), so there is no gradient
    w.r.t. it. Covers squares, square roots (p=0.5) and reciprocals (p=-1).
    """

    def __init__(self, exponent: float) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        return x.data ** self.exponent

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        return (grad_output * self.exponent * x.data ** (self.exponent - 1),)


class Exp(Function):
    """Element-wise exponential: z = exp(x)."""

    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.out = xp.exp(x.data)
        return self.out

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # d/dx exp(x) = exp(x) - reuse the forward result instead of
        # recomputing it
        return (grad_output * self.out,)


class Log(Function):
    """Element-wise natural logarithm: z = log(x)."""

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        xp = get_array_module(x.data)
        return xp.log(x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        return (grad_output / x.data,)
