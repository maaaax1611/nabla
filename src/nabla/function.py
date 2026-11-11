from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nabla.tensor import Tensor


_grad_enabled = True


def is_grad_enabled() -> bool:
    """True unless execution is currently inside a `no_grad()` block."""
    return _grad_enabled


class no_grad:
    """Context manager (and decorator) that disables graph construction.

    Inside the block every op still computes its forward result, but the
    output tensor is not connected to its inputs and never requires grad -
    so evaluation and sampling don't keep activations alive for a backward
    pass that will never happen.

        with nabla.no_grad():
            logits = model(x)
    """

    def __enter__(self) -> None:
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *exc) -> None:
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return fn(*args, **kwargs)

        return wrapper


class Function:
    """Base class for all differentiable operations.

    All custom operations should inherit from this class
    and implement ``forward`` and ``backward``.

    Attributes:
        saved_tensors: Tensors saved during forward for use in backward.
    """

    def __init__(self) -> None:
        self.saved_tensors: tuple[Tensor, ...] = ()

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Save tensors needed for the backward pass."""
        self.saved_tensors = tensors

    def forward(self, *args: Tensor) -> NDArray:
        """Compute the forward pass. Must be overridden by subclasses."""
        raise NotImplementedError

    def backward(self, grad_output: NDArray) -> tuple[NDArray, ...]:
        """Compute gradients w.r.t. inputs. Must be overridden by subclasses."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs) -> Tensor:
        """Create a new operation instance, run forward, and build the computation graph.

        Args:
            *inputs: Input tensors to the operation.
            **kwargs: Extra non-tensor configuration passed to the operation's
                constructor (e.g. ``new_shape`` for Reshape, ``axes`` for Transpose).

        Returns:
            A new Tensor containing the result with graph connections set up.
        """
        from nabla.tensor import Tensor

        ctx = cls(**kwargs)
        result = ctx.forward(*inputs)
        out = Tensor(result)

        # Only wire the result into the graph if a gradient can actually
        # flow through it. Otherwise the Function (and everything it saved
        # for backward) is dropped right here instead of being kept alive
        # by a tensor nobody will ever call backward() through.
        if _grad_enabled and any(t.requires_grad for t in inputs):
            out._ctx = ctx
            out._prev = inputs
            out.requires_grad = True

        return out
