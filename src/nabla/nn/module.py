from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nabla.backend import get_array_module, is_gpu_array, to_device
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

    def state_dict(self) -> list[NDArray]:
        """A CPU-resident snapshot of every parameter's values, in the
        same order as `parameters()` - for checkpointing (see
        `nabla/checkpoint.py`). Independent copies: mutating the
        returned arrays, or continuing to train this module, never
        changes the snapshot.
        """
        return [to_device(p.data, "cpu").copy() for p in self.parameters()]

    def load_state_dict(self, state: list[NDArray]) -> None:
        """Load parameter values from a `state_dict()` snapshot back into
        this module's parameters (matched by `parameters()` order).

        Each parameter's *current* device is preserved - loading a CPU
        snapshot into a model already moved to the GPU keeps it on the
        GPU, no separate `.to()` call needed after loading.
        """
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError(f"Expected {len(params)} parameters, got {len(state)}.")
        for param, value in zip(params, state):
            xp = get_array_module(param.data)
            param.data = xp.asarray(value)

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Tensor, **kwargs) -> Tensor:
        """Define the forward computation. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")