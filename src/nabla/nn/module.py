from __future__ import annotations

import numpy as np

from nabla.backend import is_gpu_array, to_device
from nabla.tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should implement ``forward()`` to define the computation.
    Parameters are automatically discovered from attributes.
    """

    def __init__(self) -> None:
        self.training = True

    def train(self, mode: bool = True) -> "Module":
        """Set this module and all submodules to training or evaluation mode.

        Layers like BatchNorm2D behave differently depending on this flag
        (batch statistics vs. running statistics).
        """
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
        return self

    def eval(self) -> "Module":
        """Set this module and all submodules to evaluation mode."""
        return self.train(False)

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters in this module and its submodules."""
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def to(self, device: str) -> "Module":
        """Move every parameter, buffer, and submodule to "cpu" or "cuda".

        Recognizes three kinds of attributes: `Tensor`s (parameters like
        weights/biases, moved via Tensor.to - covers both trainable
        weights and fixed-but-Tensor-wrapped buffers like
        VisionTransformer's CLS-token selector), plain ndarrays (fixed
        buffers that were never wrapped in a Tensor, like
        PositionalEncoding's sinusoidal table), and nested `Module`s
        (recursed into). Anything else (plain Python state like `self.p`
        on Dropout) is left untouched.
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                value.to(device)
            elif isinstance(value, Module):
                value.to(device)
            elif isinstance(value, np.ndarray) or is_gpu_array(value):
                setattr(self, name, to_device(value, device))
        return self

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Tensor, **kwargs) -> Tensor:
        """Define the forward computation. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")