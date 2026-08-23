from __future__ import annotations

import numpy as np

from nabla.backend import is_gpu_array
from nabla.nn.module import Module
from nabla.ops.loss import BCEWithLogitsLoss as BCEWithLogitsLossFunction
from nabla.ops.loss import DiceLoss as DiceLossFunction
from nabla.ops.loss import SoftmaxCrossEntropy
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


class CrossEntropyLoss(Module):
    """Softmax + negative log-likelihood loss for multi-class classification.

    Applies softmax to the logits internally, so the model should output
    raw, unnormalized scores (no activation on the final layer).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the cross-entropy loss between logits and class labels.

        Args:
            logits: Model output of shape (batch, num_classes), unnormalized.
            targets: Integer class labels of shape (batch,), either a Tensor
                or a plain array-like.

        Returns:
            Scalar tensor containing the mean cross-entropy loss.
        """
        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a Tensor.")
        if not isinstance(targets, Tensor):
            targets = Tensor(np.asarray(targets))
        if is_gpu_array(logits.data) and not is_gpu_array(targets.data):
            # convenience: labels commonly come straight from a CPU
            # DataLoader even when the model has been moved to the GPU -
            # match them to logits' device rather than forcing every
            # caller to remember to move targets too
            targets = targets.to("cuda")
        if logits.data.ndim != 2:
            raise ValueError(f"Expected logits of shape (batch, num_classes), got {logits.data.shape}.")
        if targets.data.shape != (logits.data.shape[0],):
            raise ValueError(
                f"Expected targets of shape ({logits.data.shape[0]},), got {targets.data.shape}."
            )
        return SoftmaxCrossEntropy.apply(logits, targets)


class BCEWithLogitsLoss(Module):
    """Binary cross-entropy computed directly from logits.

    Expects `logits` to be raw, unnormalized scores - no sigmoid applied.
    Numerically stable version of applying sigmoid then binary cross-
    entropy separately (see `ops.loss.BCEWithLogitsLoss`).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute BCE loss between logits and binary targets.

        Args:
            logits: Raw model output, shape (batch, ...).
            targets: Ground-truth binary labels, same shape as logits.

        Returns:
            Scalar tensor containing the mean BCE loss over every element.
        """
        if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both logits and targets must be instances of Tensor.")
        if logits.data.shape != targets.data.shape:
            raise ValueError(
                f"Logits and targets must have the same shape. "
                f"Got {logits.data.shape} and {targets.data.shape}."
            )
        return BCEWithLogitsLossFunction.apply(logits, targets)


class DiceLoss(Module):
    """Soft Dice loss for binary segmentation: 1 - Dice coefficient.

    Expects `predictions` to already be probabilities in [0, 1] (e.g. after
    a sigmoid) - this Function does not apply an activation itself.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the Dice loss between predicted and target masks.

        Args:
            predictions: Predicted probabilities, shape (batch, ...).
            targets: Ground-truth binary mask, same shape as predictions.

        Returns:
            Scalar tensor containing the mean Dice loss over the batch.
        """
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be instances of Tensor.")
        if predictions.data.shape != targets.data.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape. "
                f"Got {predictions.data.shape} and {targets.data.shape}."
            )
        return DiceLossFunction.apply(predictions, targets, eps=self.eps)