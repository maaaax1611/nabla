from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Sum(Function):
    """Reduce all elements to a scalar by summation."""

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        return np.sum(x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        return (np.ones_like(x.data) * grad_output,)


class Mean(Function):
    """Reduce all elements to a scalar by averaging."""

    def forward(self, x: Tensor) -> NDArray:
        self.save_for_backward(x)
        return np.mean(x.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        (x,) = self.saved_tensors
        n = x.data.size
        return (np.ones_like(x.data) * (grad_output / n),)