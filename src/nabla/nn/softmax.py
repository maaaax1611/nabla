from __future__ import annotations

from nabla.nn.module import Module
from nabla.ops.softmax import Softmax as SoftmaxFunction
from nabla.tensor import Tensor


class Softmax(Module):
    """Softmax layer. No parameters, no train/eval distinction - just wraps
    ops/softmax.py so it composes like any other layer inside a Module.

    Args:
        axis: Axis to apply softmax over (default: the last axis).
    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        return SoftmaxFunction.apply(x, axis=self.axis)
