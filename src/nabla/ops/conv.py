from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Conv2D(Function):
    """2D convolution: out = conv(x, weight) + bias, implemented via im2col.

    Shapes:
        x:      (batch, in_channels, H, W)
        weight: (out_channels, in_channels, kh, kw)
        bias:   (out_channels,)
        out:    (batch, out_channels, out_h, out_w)

    where out_h = (H + 2*padding - kh) // stride + 1 (analogous for out_w).
    """

    def __init__(self, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor, weight: Tensor, bias: Tensor) -> NDArray:
        self.save_for_backward(x, weight, bias)
        xp = get_array_module(x.data)

        if self.padding > 0:
            self.x_padded = xp.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            self.x_padded = x.data

        in_channels, kh, kw = weight.data.shape[1:]
        self.batch_size = x.data.shape[0]
        self.out_h = (x.data.shape[2] + 2 * self.padding - kh) // self.stride + 1
        self.out_w = (x.data.shape[3] + 2 * self.padding - kw) // self.stride + 1

        # extract every kh x kw patch, then keep only every `stride`-th one
        windows = xp.lib.stride_tricks.sliding_window_view(self.x_padded, (kh, kw), axis=(2, 3))
        # windows: (batch, in_channels, out_h, out_w, kh, kw)
        windows = windows[:, :, :: self.stride, :: self.stride, :, :]

        # im2col: each column holds one flattened receptive field
        self.cols = windows.transpose(1, 4, 5, 0, 2, 3).reshape(in_channels * kh * kw, -1)
        # cols: (in_channels*kh*kw, batch*out_h*out_w)

        weight_flat = weight.data.reshape(weight.data.shape[0], -1)  # (out_channels, in_channels*kh*kw)

        out_flat = weight_flat @ self.cols + bias.data.reshape(-1, 1)
        # out_flat: (out_channels, batch*out_h*out_w)

        # reshape in the current axis order (out_channels, batch, out_h, out_w)
        # first, then transpose - reshape alone cannot reorder axes
        out = out_flat.reshape(weight.data.shape[0], self.batch_size, self.out_h, self.out_w)
        return out.transpose(1, 0, 2, 3)  # (batch, out_channels, out_h, out_w)

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        # grad_output: (batch, out_channels, out_h, out_w)
        _, weight, _ = self.saved_tensors
        xp = get_array_module(grad_output)
        in_channels, kh, kw = weight.data.shape[1:]

        grad_bias = xp.sum(grad_output, axis=(0, 2, 3))  # (out_channels,)

        # undo the forward transpose to match the (out_channels, batch*out_h*out_w) layout of `cols`
        grad_output_flat = grad_output.transpose(1, 0, 2, 3).reshape(grad_output.shape[1], -1)

        weight_flat = weight.data.reshape(weight.data.shape[0], -1)  # (out_channels, in_channels*kh*kw)

        grad_weight_flat = grad_output_flat @ self.cols.T  # (out_channels, in_channels*kh*kw)
        grad_weight = grad_weight_flat.reshape(weight.data.shape)

        grad_cols = weight_flat.T @ grad_output_flat  # (in_channels*kh*kw, batch*out_h*out_w)

        # scatter each patch gradient back into its receptive field, accumulating
        # where windows overlap (stride < kernel size).
        #
        # Looping over every output position (out_h * out_w iterations) is what
        # this used to do - fine at MNIST's 28x28, but 57,600 iterations (each
        # its own tiny GPU kernel launch) at BraTS's 240x240 turned a single
        # conv layer's backward into tens of seconds. Looping over kernel
        # offsets instead (kh * kw iterations - 9 for a 3x3 kernel) does the
        # same accumulation with a strided-slice add that covers every output
        # position for that offset in one vectorized op.
        grad_x_padded = xp.zeros_like(self.x_padded)
        grad_cols = grad_cols.reshape(in_channels, kh, kw, self.batch_size, self.out_h, self.out_w)

        for di in range(kh):
            for dj in range(kw):
                # every output position's contribution at kernel offset (di, dj),
                # landing on a regularly-strided grid of input positions
                patch_grad = grad_cols[:, di, dj, :, :, :].transpose(1, 0, 2, 3)  # (batch, in_channels, out_h, out_w)
                h_end = di + self.stride * (self.out_h - 1) + 1
                w_end = dj + self.stride * (self.out_w - 1) + 1
                grad_x_padded[:, :, di:h_end:self.stride, dj:w_end:self.stride] += patch_grad

        if self.padding > 0:
            p = self.padding
            grad_x = grad_x_padded[:, :, p:-p, p:-p]
        else:
            grad_x = grad_x_padded

        return grad_x, grad_weight, grad_bias
