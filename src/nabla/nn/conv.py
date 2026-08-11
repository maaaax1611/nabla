from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from nabla.init import he
from nabla.nn.module import Module
from nabla.tensor import Tensor


class Conv2D(Module):
    """A 2D convolutional layer: y = conv2d(x, weight) + bias.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the (square) convolving kernel.
        stride: Step size of the sliding window.
        padding: Zero-padding added to both sides of the H/W axes.
        weight_init: Callable mapping a shape to an initial weight array.
            Defaults to He initialization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_init: Callable[[tuple[int, ...]], NDArray] = he,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Tensor(weight_init(weight_shape), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolution to input.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Output tensor of shape (batch, out_channels, out_h, out_w).
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {x.data.shape[1]}."
            )
        return x.conv2d(self.weight, self.bias, stride=self.stride, padding=self.padding)
