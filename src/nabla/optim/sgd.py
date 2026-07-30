from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nabla.tensor import Tensor


class SGD:
    """Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Update parameters using their gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters:
            param.grad = None