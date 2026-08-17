from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.broadcast import unbroadcast
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class MatMul(Function):
    """Matrix multiplication: z = a @ b.

    Supports batched matmul: any leading axes are treated as batch
    dimensions (broadcast against each other, exactly like np.matmul),
    and the actual matrix product happens over the trailing two axes.
    Plain 2D matrices are just the batch-free special case - np.matmul
    behaves identically to np.dot there.
    """

    def forward(self, a: Tensor, b: Tensor) -> NDArray:
        self.save_for_backward(a, b)
        xp = get_array_module(a.data, b.data)
        return xp.matmul(a.data, b.data)

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        a, b = self.saved_tensors
        xp = get_array_module(grad_output)
        # batched matmul backward: swap the trailing two axes only, never
        # the batch axes, then reduce away any axes that were broadcast
        # (e.g. a 2D weight matrix shared across a batched input)
        grad_a = xp.matmul(grad_output, xp.swapaxes(b.data, -1, -2))
        grad_b = xp.matmul(xp.swapaxes(a.data, -1, -2), grad_output)
        return unbroadcast(grad_a, a.data.shape), unbroadcast(grad_b, b.data.shape)

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
        xp = get_array_module(x.data)
        return xp.transpose(x.data, self.axes)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        xp = get_array_module(grad_output)
        if self.axes is None:
            # reversing all axes is its own inverse
            return (xp.transpose(grad_output),)
        # recover the inverse permutation to undo the forward transpose -
        # self.axes is a plain Python tuple of axis indices, not
        # device data, so this is always a small numpy computation
        # regardless of which backend grad_output lives on
        inverse_axes = np.argsort(self.axes)
        return (xp.transpose(grad_output, inverse_axes),)