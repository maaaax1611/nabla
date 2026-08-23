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


class BCEWithLogitsLoss(Function):
    """Binary cross-entropy computed directly from logits.

    Shapes:
        logits:  (batch, ...) raw, unnormalized scores (no sigmoid applied)
        targets: (batch, ...) same shape, binary labels (0.0 or 1.0)
        out:     scalar, mean loss over every element

    Fusing sigmoid + log into one Function (rather than calling
    `.sigmoid()` then `log()` separately) avoids ever computing log(0):
    a confidently wrong prediction can push sigmoid(logits) to exactly
    0.0 or 1.0 in float32, and log of that is -inf. The stable
    formulation below is mathematically identical to
    -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))] but never evaluates
    sigmoid or log on values that could over/underflow:

        loss = max(x, 0) - x*y + log(1 + exp(-|x|))

    and gives an equally simple backward pass to SoftmaxCrossEntropy's:
    d(loss)/d(logits) = (sigmoid(logits) - targets) / num_elements.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> NDArray:
        self.save_for_backward(logits, targets)
        xp = get_array_module(logits.data)
        x, y = logits.data, targets.data

        loss = xp.maximum(x, 0) - x * y + xp.log1p(xp.exp(-xp.abs(x)))
        self.num_elements = x.size
        return loss.mean()

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        logits, targets = self.saved_tensors
        xp = get_array_module(grad_output)

        sigmoid_x = 1 / (1 + xp.exp(-logits.data))
        grad_logits = (sigmoid_x - targets.data) * (grad_output / self.num_elements)

        # targets are fixed 0/1 labels, never differentiable
        grad_targets = xp.zeros_like(targets.data)
        return grad_logits, grad_targets


class DiceLoss(Function):
    """Soft Dice loss: 1 - Dice coefficient, for binary segmentation.

    Shapes:
        probs:  (batch, ...) predicted probabilities in [0, 1] (already
                sigmoid'd - this Function does not apply an activation)
        target: (batch, ...) same shape, ground-truth mask (0.0 or 1.0)
        out:    scalar, mean loss over the batch
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: Tensor, target: Tensor) -> NDArray:
        xp = get_array_module(probs.data)
        self.save_for_backward(probs, target)
        # calculate intersection for every axis except the batch axis (0)
        self.intersection = (probs.data * target.data).sum(axis=tuple(range(1, probs.data.ndim)))
        # same for union
        self.union = (
            xp.sum(probs.data, axis=tuple(range(1, probs.data.ndim)))
            + xp.sum(target.data, axis=tuple(range(1, target.data.ndim)))
        )
        dice = (2 * self.intersection + self.eps) / (self.union + self.eps)
        loss = 1 - dice
        return loss.mean()

    def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
        p_tensor, y_tensor = self.saved_tensors
        p, y = p_tensor.data, y_tensor.data
        I, U, eps = self.intersection, self.union, self.eps
        xp = get_array_module(p)

        batch = p.shape[0]
        shape = (batch,) + (1,) * (p.ndim - 1)
        I = I.reshape(shape)
        U = U.reshape(shape)

        numerator = (2 * I + eps) - 2 * y * (U + eps)
        grad_p = numerator / (U + eps) ** 2 / batch

        grad_target = xp.zeros_like(y)
        return grad_output * grad_p, grad_target