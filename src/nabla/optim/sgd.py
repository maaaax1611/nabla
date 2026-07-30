from __future__ import annotations

from nabla.optim.optimizer import Optimizer
from nabla.tensor import Tensor


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        """Update parameters using their gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad