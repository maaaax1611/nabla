from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class MaxPool2D(Function):
    """2D max pooling: out[b, c, i, j] = max over the (kh, kw) window at (i, j).

    Shapes:
        x:   (batch, channels, H, W)
        out: (batch, channels, out_h, out_w)

    where out_h = (H - kernel_size) // stride + 1 (analogous for out_w).
    Unlike Conv2D there are no weights/bias and channels are left untouched
    (no "out_channels").
    """

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # mirrors PyTorch: stride=None -> stride = kernel_size (non-overlapping windows)
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.input_shape = x.data.shape
        self.out_h = (x.data.shape[2] - self.kernel_size) // self.stride + 1
        self.out_w = (x.data.shape[3] - self.kernel_size) // self.stride + 1

        # extract every kh x kw patch, then keep only every `stride`-th one
        windows = xp.lib.stride_tricks.sliding_window_view(x.data, (self.kernel_size, self.kernel_size), axis=(2, 3))
        # windows: (batch, channels, out_h, out_w, kh, kw)
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]

        # flatten the kernel window into a single axis to reduce over
        windows = windows.reshape(*windows.shape[:4], -1)  # (batch, channels, out_h, out_w, kh*kw)

        # remember which position in each window held the max, needed to route
        # the gradient back to exactly that position in backward
        self.argmax = xp.argmax(windows, axis=-1)  # (batch, channels, out_h, out_w)
        return xp.max(windows, axis=-1)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # grad_output: (batch, channels, out_h, out_w)
        #
        # Looping over every output position (out_h * out_w iterations, each
        # its own xp.add.at call) doesn't scale to BraTS-sized inputs - see
        # the same fix in Conv2D.backward. Every output position's target
        # input index can be computed for the whole (batch, channels, out_h,
        # out_w) grid in one shot instead, so a single xp.add.at call handles
        # the entire scatter (still accumulating correctly if windows overlap).
        xp = get_array_module(grad_output)
        batch, channels, out_h, out_w = grad_output.shape
        grad_x = xp.zeros(self.input_shape, dtype=grad_output.dtype)

        # di, dj = divmod(argmax, kernel_size): unravel_index's generic
        # N-dimensional-shape handling is serious overhead for what is,
        # for a fixed 2D kernel_size, just integer division and remainder
        di, dj = xp.divmod(self.argmax, self.kernel_size)  # each (batch, channels, out_h, out_w)

        batch_idx = xp.arange(batch).reshape(batch, 1, 1, 1)
        channel_idx = xp.arange(channels).reshape(1, channels, 1, 1)
        i_idx = xp.arange(out_h).reshape(1, 1, out_h, 1)
        j_idx = xp.arange(out_w).reshape(1, 1, 1, out_w)

        h_idx = i_idx * self.stride + di
        w_idx = j_idx * self.stride + dj

        xp.add.at(grad_x, (batch_idx, channel_idx, h_idx, w_idx), grad_output)

        return (grad_x,)


class AvgPool2D(Function):
    """2D average pooling: out[b, c, i, j] = mean over the (kh, kw) window at (i, j).

    Simpler than MaxPool2D: the backward gradient is just spread evenly over
    every position in the window, no argmax bookkeeping needed.
    """

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.input_shape = x.data.shape
        self.out_h = (x.data.shape[2] - self.kernel_size) // self.stride + 1
        self.out_w = (x.data.shape[3] - self.kernel_size) // self.stride + 1

        windows = xp.lib.stride_tricks.sliding_window_view(x.data, (self.kernel_size, self.kernel_size), axis=(2, 3))
        # windows: (batch, channels, out_h, out_w, kh, kw)
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]
        windows = windows.reshape(*windows.shape[:4], -1)  # (batch, channels, out_h, out_w, kh*kw)

        return xp.mean(windows, axis=-1)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # grad_output: (batch, channels, out_h, out_w)
        xp = get_array_module(grad_output)
        grad_x = xp.zeros(self.input_shape)

        for i in range(self.out_h):
            for j in range(self.out_w):
                grad_per_position = grad_output[:, :, i, j] / (self.kernel_size * self.kernel_size)
                h0, w0 = i * self.stride, j * self.stride
                grad_x[:, :, h0 : h0 + self.kernel_size, w0 : w0 + self.kernel_size] += grad_per_position[:, :, None, None]

        return (grad_x,)
