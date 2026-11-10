from __future__ import annotations

import math
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


class Tanh(Function):
    """Tanh activation"""
    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.out = xp.tanh(x.data)
        return self.out

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        return (grad_output * (1 - self.out ** 2),)


class GELU(Function):
    """GELU activation, tanh approximation (the variant GPT-2 and BERT use):

        gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    The exact form needs the Gaussian CDF (erf), which NumPy doesn't ship;
    the approximation is within ~1e-3 of it everywhere.
    """

    _C = math.sqrt(2.0 / math.pi)
    _A = 0.044715

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        xp = get_array_module(x.data)
        self.t = xp.tanh(self._C * (x.data + self._A * x.data ** 3))
        return 0.5 * x.data * (1 + self.t)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        # product rule on 0.5 * x * (1 + t), with dt/dx = (1 - t^2) * u'(x)
        du = self._C * (1 + 3 * self._A * x.data ** 2)
        local = 0.5 * (1 + self.t) + 0.5 * x.data * (1 - self.t ** 2) * du
        return (grad_output * local,)
