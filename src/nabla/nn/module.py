from __future__ import annotations

from nabla.tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should implement ``forward()`` to define the computation.
    Parameters are automatically discovered from attributes.
    """

    def __init__(self) -> None:
        pass

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters in this module and its submodules."""
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args: Tensor) -> Tensor:
        return self.forward(*args)

    def forward(self, *args: Tensor) -> Tensor:
        """Define the forward computation. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")