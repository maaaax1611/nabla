from __future__ import annotations

from nabla.optim.optimizer import Optimizer
from nabla.regularizers import Regularizer
from nabla.tensor import Tensor


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
        regularizers: Optional list of regularizers (e.g. L1/L2 weight decay).
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        regularizers: list[Regularizer] | None = None,
    ) -> None:
        super().__init__(parameters, lr, regularizers)

    def step(self) -> None:
        """Update parameters using their gradients."""
        for param in self.parameters:
            if param.grad is not None:
                grad = self._regularized_grad(param)
                param.data -= self.lr * grad