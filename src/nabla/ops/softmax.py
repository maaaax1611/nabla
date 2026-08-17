from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class Softmax(Function):
    """Standalone softmax, differentiable on its own (unlike the fused
    SoftmaxCrossEntropy in ops/loss.py, which never needs to backprop
    through softmax by itself - see docs/05-softmax-cross-entropy.md for
    why that fusion collapses the gradient so much).

    Shapes:
        x:   any, softmax is applied over `axis` (default: the last axis -
             e.g. the class/sequence axis for attention scores)
        out: same shape as x

    softmax(x)_k = exp(x_k) / sum_j exp(x_j), over the chosen axis.
    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> NDArray:
        # log-sum-exp trick: subtract the max before exponentiating so
        # exp() never overflows, without changing the softmax result
        xp = get_array_module(x.data)
        shifted = x.data - x.data.max(axis=self.axis, keepdims=True)
        exp_shifted = xp.exp(shifted)
        sum_exp = exp_shifted.sum(axis=self.axis, keepdims=True)
        self.probs = exp_shifted / sum_exp
        return self.probs

    def backward(self, grad_output: NDArray) -> tuple[NDArray]:
        # softmax's own Jacobian is dp_j/dx_k = p_j * (delta_jk - p_k);
        # summing grad_output_j * dp_j/dx_k over j collapses to this
        # Jacobian-vector product (see the derivation in the docs):
        #   dL/dx_k = p_k * (dL/dout_k - sum_j dL/dout_j * p_j)
        xp = get_array_module(grad_output)
        weighted_sum = xp.sum(grad_output * self.probs, axis=self.axis, keepdims=True)
        return (self.probs * (grad_output - weighted_sum),)