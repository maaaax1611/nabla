from __future__ import annotations

from nabla.nn.module import Module
from nabla.ops.dropout import Dropout as DropoutFunction
from nabla.tensor import Tensor


class Dropout(Module):
    """Dropout layer.

    In training mode, randomly zeroes elements with probability p (see
    ``ops.dropout.Dropout``). In eval mode, it's a no-op — the scaling that
    would otherwise be needed at eval time already happened during training
    ("inverted" dropout), so x is passed through unchanged.

    Args:
        p: Probability of zeroing an element during training.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        if self.training:
            return DropoutFunction.apply(x, p=self.p)
        return x
