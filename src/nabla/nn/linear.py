from __future__ import annotations

import numpy as np

from nabla.nn.module import Module
from nabla.tensor import Tensor


class Linear(Module):
    """A fully connected linear layer: y = x @ weight + bias.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # He initialization (suitable for ReLU activations)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transformation to input.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input with last dimension {self.in_features}, "
                f"but got {x.data.shape[-1]}."
            )
        return x.matmul(self.weight) + self.bias