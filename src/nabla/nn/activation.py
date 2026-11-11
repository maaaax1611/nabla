from __future__ import annotations

from nabla.nn.module import Module
from nabla.tensor import Tensor


class ReLU(Module):
    """ReLU activation as a layer: max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation as a layer: 1 / (1 + exp(-x))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Tanh activation as a layer."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """GELU activation as a layer (tanh approximation)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()
