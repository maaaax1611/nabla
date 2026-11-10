from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor

Axis = int | tuple[int, ...] | None


def _expand_reduced(grad: NDArray, shape: tuple[int, ...], axis: Axis, keepdims: bool) -> NDArray:
    """Undo a reduction's shape change: put the reduced axes back as
    size-1 axes (unless keepdims already left them there), then broadcast
    to the input's full shape."""
    xp = get_array_module(grad)
    grad = xp.asarray(grad)
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, int) else axis
        # normalize negatives against the *input* rank, then insert in
        # ascending order so earlier insertions don't shift later ones
        for ax in sorted(a % len(shape) for a in axes):
            grad = xp.expand_dims(grad, ax)
    return xp.broadcast_to(grad, shape)


class Sum(Function):
    """Sum over `axis` (all elements if axis is None).

    Every input element contributes to its output element with weight 1,
    so backward just copies the incoming gradient to every position that
    was summed into it.
    """

    def __init__(self, axis: Axis = None, keepdims: bool = False) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> NDArray:
        self.input_shape = x.data.shape
        xp = get_array_module(x.data)
        return xp.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        grad = _expand_reduced(grad_output, self.input_shape, self.axis, self.keepdims)
        return (grad.copy(),)


class Mean(Function):
    """Mean over `axis` (all elements if axis is None).

    Same as Sum, except each element's weight is 1/n, where n is the
    number of elements averaged into one output element.
    """

    def __init__(self, axis: Axis = None, keepdims: bool = False) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> NDArray:
        self.input_shape = x.data.shape
        xp = get_array_module(x.data)
        out = xp.mean(x.data, axis=self.axis, keepdims=self.keepdims)
        self.n = x.data.size // max(out.size, 1)
        return out

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        grad = _expand_reduced(grad_output, self.input_shape, self.axis, self.keepdims)
        return (grad / self.n,)
