from __future__ import annotations

from nabla.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    Subclasses should implement ``step()`` to define how parameters
    are updated using their gradients.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
    """

    def __init__(self, parameters: list[Tensor], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Update parameters using their gradients. Must be overridden by subclasses."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters:
            param.grad = None
