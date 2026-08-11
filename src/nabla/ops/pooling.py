from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

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
        self.input_shape = x.data.shape
        self.out_h = (x.data.shape[2] - self.kernel_size) // self.stride + 1
        self.out_w = (x.data.shape[3] - self.kernel_size) // self.stride + 1

        # extract every kh x kw patch, then keep only every `stride`-th one
        windows = sliding_window_view(x.data, (self.kernel_size, self.kernel_size), axis=(2, 3))
        # windows: (batch, channels, out_h, out_w, kh, kw)
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]

        # flatten the kernel window into a single axis to reduce over
        windows = windows.reshape(*windows.shape[:4], -1)  # (batch, channels, out_h, out_w, kh*kw)

        # remember which position in each window held the max, needed to route
        # the gradient back to exactly that position in backward
        self.argmax = np.argmax(windows, axis=-1)  # (batch, channels, out_h, out_w)
        return np.max(windows, axis=-1)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # grad_output: (batch, channels, out_h, out_w)
        grad_x = np.zeros(self.input_shape, dtype=grad_output.dtype)

        batch_idx = np.arange(self.input_shape[0])[:, None]  # (batch, 1)
        channel_idx = np.arange(self.input_shape[1])[None, :]  # (1, channels)

        # scatter each output position's gradient back to the input position
        # that produced the max, accumulating where windows overlap
        for i in range(self.out_h):
            for j in range(self.out_w):
                di, dj = np.unravel_index(self.argmax[:, :, i, j], (self.kernel_size, self.kernel_size))
                h_idx = i * self.stride + di
                w_idx = j * self.stride + dj
                np.add.at(grad_x, (batch_idx, channel_idx, h_idx, w_idx), grad_output[:, :, i, j])

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
        self.input_shape = x.data.shape
        self.out_h = (x.data.shape[2] - self.kernel_size) // self.stride + 1
        self.out_w = (x.data.shape[3] - self.kernel_size) // self.stride + 1

        windows = sliding_window_view(x.data, (self.kernel_size, self.kernel_size), axis=(2, 3))
        # windows: (batch, channels, out_h, out_w, kh, kw)
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]
        windows = windows.reshape(*windows.shape[:4], -1)  # (batch, channels, out_h, out_w, kh*kw)

        return np.mean(windows, axis=-1)

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # grad_output: (batch, channels, out_h, out_w)
        grad_x = np.zeros(self.input_shape)

        for i in range(self.out_h):
            for j in range(self.out_w):
                grad_per_position = grad_output[:, :, i, j] / (self.kernel_size * self.kernel_size)
                h0, w0 = i * self.stride, j * self.stride
                grad_x[:, :, h0 : h0 + self.kernel_size, w0 : w0 + self.kernel_size] += grad_per_position[:, :, None, None]

        return (grad_x,)
