from __future__ import annotations

from nabla.nn.module import Module
from nabla.tensor import Tensor


class Upsample(Module):
    """Nearest-neighbor upsampling layer.

    No learnable parameters - each pixel is duplicated into a scale x scale
    block. See ``ops.upsample.Upsample`` for the forward/backward math.

    Args:
        scale: Integer factor to upsample H and W by.
    """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        return x.upsample(scale=self.scale)
