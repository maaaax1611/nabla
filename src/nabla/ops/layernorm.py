from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class LayerNorm(Function):
    """Layer normalization: normalizes each sample independently over its
    last (feature) axis, unlike BatchNorm2D which normalizes per channel
    over the batch (+ spatial axes).

    Shapes:
        x:     (*, features) - any number of leading axes, last axis gets
               normalized
        gamma: (features,)
        beta:  (features,)
        out:   same shape as x

    mean/var are computed per sample (i.e. per "row", over the last axis) -
    unlike BatchNorm2D, normalizing one sample doesn't depend on any other
    sample in the batch. That's also why there's no train/eval distinction
    or running stats here: the op behaves identically on every call.

    Since gamma/beta already have shape (features,) and features is the
    *last* axis of x, they broadcast directly against x with no reshape
    (unlike BatchNorm2D, which needs (1, C, 1, 1) because channels sit on
    the second axis there).
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> NDArray:
        self.save_for_backward(gamma, beta)
        xp = get_array_module(x.data)

        mean = xp.mean(x.data, axis=-1, keepdims=True)
        self.var = xp.var(x.data, axis=-1, keepdims=True, ddof=0)

        self.x_centered = x.data - mean
        self.x_hat = self.x_centered / xp.sqrt(self.var + self.eps)

        self.N = x.data.shape[-1]  # elements per normalized row

        return gamma.data * self.x_hat + beta.data

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        gamma, _ = self.saved_tensors
        xp = get_array_module(grad_output)

        # sum over every axis except the last (normalized) one
        leading_axes = tuple(range(grad_output.ndim - 1))
        grad_beta = xp.sum(grad_output, axis=leading_axes)
        grad_gamma = xp.sum(grad_output * self.x_hat, axis=leading_axes)

        grad_x_hat = grad_output * gamma.data

        # same three-path chain rule as BatchNorm2D, over axis=-1 instead
        # of (0, 2, 3), with self.N = features instead of batch*H*W
        grad_var = xp.sum(
            grad_x_hat * self.x_centered * -0.5 * (self.var + self.eps) ** -1.5,
            axis=-1,
            keepdims=True,
        )
        grad_mean = xp.sum(
            grad_x_hat * -1 / xp.sqrt(self.var + self.eps),
            axis=-1,
            keepdims=True
        )
        grad_mean += grad_var * xp.sum(-2 * self.x_centered, axis=-1, keepdims=True) / self.N
        grad_x = grad_x_hat / xp.sqrt(self.var + self.eps)
        grad_x += grad_var * 2 * self.x_centered / self.N
        grad_x += grad_mean / self.N

        return grad_x, grad_gamma, grad_beta
