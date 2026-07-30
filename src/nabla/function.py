from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nabla.tensor import Tensor


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
    def apply(cls, *inputs: Tensor) -> Tensor:
        """Create a new operation instance, run forward, and build the computation graph.

        Args:
            *inputs: Input tensors to the operation.

        Returns:
            A new Tensor containing the result with graph connections set up.
        """
        from nabla.tensor import Tensor

        ctx = cls()
        result = ctx.forward(*inputs)
        out = Tensor(result)

        out._ctx = ctx
        out._prev = inputs

        if any(t.requires_grad for t in inputs):
            out.requires_grad = True

        return out
