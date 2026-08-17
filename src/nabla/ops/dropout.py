from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Dropout(Function):
    """Inverted dropout: randomly zeroes elements with probability p, scales
    the survivors by 1/(1-p) so the expected activation stays unchanged.

    Shapes:
        x:   any (elementwise op, like ReLU/Sigmoid)
        out: same shape as x

    Training-only. Eval mode is a plain passthrough handled by nn.Dropout,
    not by this Function - forward() here never needs to distinguish
    train/eval itself.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p  # probability of zeroing an element (not the keep-probability)

    def forward(self, x: Tensor) -> NDArray:
        # survive with probability (1 - p), then scale survivors by
        # 1/(1-p) so E[out] == x regardless of p (inverted dropout)
        xp = get_array_module(x.data)
        keep = xp.random.rand(*x.data.shape) < (1 - self.p)
        self.mask = keep.astype(x.data.dtype) / (1 - self.p)
        return x.data * self.mask

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # out = x * mask is a plain elementwise multiply, same shape as ReLU's masking
        return (grad_output * self.mask,)
