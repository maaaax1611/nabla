from __future__ import annotations

import numpy as np

from nabla.nn.module import Module
from nabla.ops.layernorm import LayerNorm as LayerNormFunction
from nabla.tensor import Tensor


class LayerNorm(Module):
    """Layer normalization layer.

    Normalizes each sample independently over its last (feature) axis,
    using that sample's own mean/variance - unlike BatchNorm2D, there's no
    dependency on other samples in the batch, so no running stats and no
    train/eval distinction.

    Args:
        num_features: Size of the last axis of the input.
        eps: Small constant added to the variance for numerical stability.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected input with last dimension {self.num_features}, but got {x.data.shape[-1]}."
            )

        return LayerNormFunction.apply(x, self.gamma, self.beta, eps=self.eps)
