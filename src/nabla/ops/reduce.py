from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Sum(Function):
    """Reduce all elements to a scalar by summation."""

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        xp = get_array_module(x.data)
        return xp.sum(x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        xp = get_array_module(x.data)
        return (xp.ones_like(x.data) * grad_output,)


class Mean(Function):
    """Reduce all elements to a scalar by averaging."""

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        xp = get_array_module(x.data)
        return xp.mean(x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        xp = get_array_module(x.data)
        n = x.data.size
        return (xp.ones_like(x.data) * (grad_output / n),)
