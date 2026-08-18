from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Slice(Function):
    """Basic indexing: out = x[key].

    `key` is whatever gets passed to Tensor.__getitem__ - an int, a
    slice, a tuple of ints/slices/Ellipsis/None, exactly like plain
    numpy indexing (e.g. `x[:, 0]`, `x[1:3]`, `x[..., :2]`). This is
    "basic indexing" only (no boolean/fancy-array indices) - every
    output element maps back to exactly one input element, unlike
    Embedding's `table[indices]`, where the same row can be read
    multiple times.

    Shapes:
        x:   arbitrary shape.
        out: whatever `x.data[key]` produces - can drop axes (e.g. an
             int index) or shrink them (e.g. a slice).
    """

    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    def forward(self, x: Tensor) -> NDArray:
        self.original_shape = x.data.shape
        return x.data[self.key]

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # Basic indexing never reads the same source element twice, so a
        # plain assignment (not Embedding-style xp.add.at accumulation) is
        # enough to undo it: scatter grad_output back at self.key into a
        # zero array shaped like the original input.
        xp = get_array_module(grad_output)
        grad_x = xp.zeros(self.original_shape, dtype=grad_output.dtype)
        grad_x[self.key] = grad_output
        return (grad_x,)

