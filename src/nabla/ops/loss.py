from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nabla.backend import get_array_module
from nabla.function import Function

if TYPE_CHECKING:
    from nabla.tensor import Tensor


class SoftmaxCrossEntropy(Function):
    """Combined log-softmax + negative log-likelihood loss.

    Shapes:
        logits:  (batch, num_classes)
        targets: (batch,) integer class indices in [0, num_classes)
        out:     scalar, mean loss over the batch

    Computing softmax and cross-entropy in one Function (instead of a
    separate softmax op followed by a log + gather) keeps this numerically
    stable (log-sum-exp trick) and gives a very simple backward pass:
    d(loss)/d(logits) = (softmax(logits) - one_hot(targets)) / batch_size.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> NDArray:
        self.save_for_backward(logits, targets)
        xp = get_array_module(logits.data)

        # log-sum-exp trick: subtract the row max before exponentiating so
        # exp() never overflows, without changing the softmax result
        shifted = logits.data - logits.data.max(axis=1, keepdims=True)
        exp_shifted = xp.exp(shifted)
        sum_exp = exp_shifted.sum(axis=1, keepdims=True)

        self.probs = exp_shifted / sum_exp  # (batch, num_classes)
        self.batch_size = logits.data.shape[0]
        self.target_idx = targets.data.astype(xp.intp)

        log_probs = shifted - xp.log(sum_exp)  # (batch, num_classes)
        picked = log_probs[xp.arange(self.batch_size), self.target_idx]
        return xp.array(-picked.mean())

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        _, targets = self.saved_tensors
        xp = get_array_module(grad_output)

        grad_logits = self.probs.copy()
        grad_logits[xp.arange(self.batch_size), self.target_idx] -= 1
        grad_logits *= grad_output / self.batch_size

        # targets are integer labels, never differentiable
        grad_targets = xp.zeros_like(targets.data, dtype=float)
        return grad_logits, grad_targets
