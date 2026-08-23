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

    def _buffer_locations(self) -> list[tuple["Module", str]]:
        """(owner, attr_name) for every plain-ndarray buffer (e.g.
        BatchNorm2D's running_mean/running_var), in traversal order -
        the same attribute-scanning convention `to()` uses to find them.
        Yielding the *location* rather than the value lets callers both
        read the current buffers (`buffers()`) and overwrite them in
        lockstep (`load_state_dict()`), matched up the same way
        `parameters()` already does for Tensors.
        """
        locations: list[tuple[Module, str]] = []
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                locations.extend(value._buffer_locations())
            elif isinstance(value, np.ndarray) or is_gpu_array(value):
                locations.append((self, name))
        return locations

    def buffers(self) -> list[NDArray]:
        """Non-trainable persistent state (e.g. BatchNorm2D's
        running_mean/running_var) that must survive a checkpoint
        round-trip but isn't learned via gradients, so it's not part of
        `parameters()`.
        """
        return [getattr(owner, name) for owner, name in self._buffer_locations()]

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

    def state_dict(self) -> dict[str, list[NDArray]]:
        """A CPU-resident snapshot of every parameter's and buffer's
        values, in the same order as `parameters()`/`buffers()` - for
        checkpointing (see `nabla/checkpoint.py`). Independent copies:
        mutating the returned arrays, or continuing to train this
        module, never changes the snapshot.

        Buffers (e.g. BatchNorm2D's running_mean/running_var) are saved
        alongside parameters, not just parameters - without them, eval
        mode after loading a checkpoint would normalize with the
        construction-time defaults (running_mean=0, running_var=1)
        instead of the statistics actually learned during training.
        """
        return {
            "params": [to_device(p.data, "cpu").copy() for p in self.parameters()],
            "buffers": [to_device(b, "cpu").copy() for b in self.buffers()],
        }

    def load_state_dict(self, state: dict[str, list[NDArray]]) -> None:
        """Load parameter and buffer values from a `state_dict()`
        snapshot back into this module (matched by `parameters()`/
        `buffers()` order).

        Each value's *current* device is preserved - loading a CPU
        snapshot into a model already moved to the GPU keeps it on the
        GPU, no separate `.to()` call needed after loading.
        """
        params = self.parameters()
        param_values = state["params"]
        if len(params) != len(param_values):
            raise ValueError(f"Expected {len(params)} parameters, got {len(param_values)}.")
        for param, value in zip(params, param_values):
            xp = get_array_module(param.data)
            param.data = xp.asarray(value)

        buffer_locations = self._buffer_locations()
        buffer_values = state["buffers"]
        if len(buffer_locations) != len(buffer_values):
            raise ValueError(f"Expected {len(buffer_locations)} buffers, got {len(buffer_values)}.")
        for (owner, name), value in zip(buffer_locations, buffer_values):
            xp = get_array_module(getattr(owner, name))
            setattr(owner, name, xp.asarray(value))

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None."""
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Tensor, **kwargs) -> Tensor:
        """Define the forward computation. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")