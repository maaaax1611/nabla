from __future__ import annotations

from numpy.typing import NDArray

from nabla.regularizers import Regularizer
from nabla.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    Subclasses should implement ``step()`` to define how parameters
    are updated using their gradients.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
        regularizers: Optional list of regularizers applied to every
            parameter's gradient before the update (e.g. L1/L2 weight decay).
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float,
        regularizers: list[Regularizer] | None = None,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.regularizers = regularizers or []

    def _regularized_grad(self, param: Tensor) -> NDArray:
        """Return param.grad with all regularizer contributions added.

        Does not mutate ``param.grad`` itself, so the raw gradient
        stays available for inspection/logging.
        """
        grad = param.grad
        for reg in self.regularizers:
            grad = grad + reg.apply(param)
        return grad

    def step(self) -> None:
        """Update parameters using their gradients. Must be overridden by subclasses."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters:
            param.grad = None
