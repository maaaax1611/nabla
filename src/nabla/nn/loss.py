from __future__ import annotations

from nabla.nn.module import Module
from nabla.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error loss"""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSE loss between predictions and targets.

        Args:
            predictions: Model output tensor.
            targets: Ground truth tensor (same shape as predictions).

        Returns:
            Scalar tensor containing the mean squared error.
        """
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be instances of Tensor.")
        if predictions.data.shape != targets.data.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got {predictions.data.shape} and {targets.data.shape}."
            )
        diff = predictions - targets
        return (diff * diff).mean()