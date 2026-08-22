from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Upsample(Function):
    """Nearest-neighbor upsampling by an integer scale factor.

    Shapes:
        x:   (batch, channels, H, W)
        out: (batch, channels, H * scale, W * scale)

    Forward: duplicate each input pixel into a scale x scale square in the output.
    Backward: sum the gradients of each scale x scale square in the output to
    get the gradient for the corresponding input pixel.
    """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> NDArray:
        xp = get_array_module(x.data)
        self.input_shape = x.data.shape

        # duplicate each pixel in the H and W dimensions by 
        # a scale factor
        out_h = xp.repeat(x.data, self.scale, axis=2)
        out = xp.repeat(out_h, self.scale, axis=3)
        return out

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        xp = get_array_module(grad_output)
        b, c, h, w = self.input_shape
        # reshape grad_output to (b, c, h, scale, w, scale) so that we can sum over the scale dimensions
        grad_reshaped = grad_output.reshape(b, c, h, self.scale, w, self.scale)
        # sum over the scale dimensions (axis 3 and 5)
        grad_input = grad_reshaped.sum(axis=(3, 5))
        return (grad_input,)