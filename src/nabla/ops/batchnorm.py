from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class BatchNorm2D(Function):
    """2D batch normalization using per-batch statistics (training mode).

    out = gamma * (x - mean) / sqrt(var + eps) + beta

    Shapes:
        x:     (batch, channels, H, W)
        gamma: (channels,)
        beta:  (channels,)
        out:   (batch, channels, H, W)

    mean/var are computed per channel, over (batch, H, W) - each channel is
    normalized independently of the others. Unlike Conv2D/Pooling, the
    shape doesn't change here.

    Note: this is "just" the differentiable normalization using batch
    statistics (training mode). Tracking running_mean/running_var for eval
    mode happens separately in the nn layer, not in this Function.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> NDArray:
        self.save_for_backward(gamma, beta)
        xp = get_array_module(x.data)

        self.batch_mean = xp.mean(x.data, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
        self.batch_var = xp.var(x.data, axis=(0, 2, 3), keepdims=True, ddof=0)  # (1, C, 1, 1)

        self.x_centered = x.data - self.batch_mean  # (batch, C, H, W)
        self.x_hat = self.x_centered / xp.sqrt(self.batch_var + self.eps)  # (batch, C, H, W)

        batch, _, H, W = x.data.shape
        self.N = batch * H * W  # elements per channel, reduced over above

        gamma_reshaped = gamma.data.reshape(1, -1, 1, 1)
        beta_reshaped = beta.data.reshape(1, -1, 1, 1)
        return gamma_reshaped * self.x_hat + beta_reshaped

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        # grad_output: (batch, channels, H, W)
        gamma, _ = self.saved_tensors
        xp = get_array_module(grad_output)

        grad_beta = xp.sum(grad_output, axis=(0, 2, 3))  # (channels,)
        grad_gamma = xp.sum(grad_output * self.x_hat, axis=(0, 2, 3))  # (channels,)

        gamma_reshaped = gamma.data.reshape(1, -1, 1, 1)
        grad_x_hat = grad_output * gamma_reshaped  # (batch, C, H, W)

        # chain rule through the normalization: x_hat depends on batch_var
        # and batch_mean, and batch_var itself depends on batch_mean, so
        # grad_mean picks up contributions from both paths
        grad_var = xp.sum(
            grad_x_hat * self.x_centered * -0.5 * (self.batch_var + self.eps) ** -1.5,
            axis=(0, 2, 3),
            keepdims=True,
        )
        grad_mean = xp.sum(grad_x_hat * -1 / xp.sqrt(self.batch_var + self.eps), axis=(0, 2, 3), keepdims=True)
        grad_mean += grad_var * xp.mean(-2 * self.x_centered, axis=(0, 2, 3), keepdims=True)

        grad_x = grad_x_hat / xp.sqrt(self.batch_var + self.eps)
        grad_x += grad_var * 2 * self.x_centered / self.N
        grad_x += grad_mean / self.N

        return grad_x, grad_gamma, grad_beta
