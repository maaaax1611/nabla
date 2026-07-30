from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nabla.tensor import Tensor


class Regularizer:
    """Base class for weight regularizers.

    Subclasses compute the gradient contribution of a penalty term
    (e.g. L1/L2) that can be added to a parameter's gradient during
    optimization, or reused elsewhere (e.g. in loss functions).
    """

    def apply(self, param: Tensor) -> NDArray:
        """Return the gradient contribution of the regularization penalty."""
        raise NotImplementedError


class L1Regularizer(Regularizer):
    """L1 regularization: penalty = weight_decay * |param|."""

    def __init__(self, weight_decay: float) -> None:
        self.weight_decay = weight_decay

    def apply(self, param: Tensor) -> NDArray:
        return self.weight_decay * np.sign(param.data)


class L2Regularizer(Regularizer):
    """L2 regularization: penalty = weight_decay/2 * param^2."""

    def __init__(self, weight_decay: float) -> None:
        self.weight_decay = weight_decay

    def apply(self, param: Tensor) -> NDArray:
        return self.weight_decay * param.data