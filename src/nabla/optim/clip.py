from __future__ import annotations

from nabla.backend import get_array_module
from nabla.tensor import Tensor


def clip_grad_norm_(parameters: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Clip gradients in-place so their combined (global) L2 norm never
    exceeds `max_norm` - mirrors PyTorch's `nn.utils.clip_grad_norm_`.

    Unlike clipping each parameter's gradient independently, this computes
    one norm over *every* parameter's gradient combined (as if they were
    concatenated into a single vector) and scales all of them down by the
    same factor if that combined norm exceeds max_norm. That keeps the
    *direction* of the overall gradient unchanged - only its magnitude is
    reduced - which is what makes this useful against exploding gradients
    in deep/recurrent-like stacks (e.g. many stacked TransformerBlocks)
    without distorting which parameters move relative to each other.

    Args:
        parameters: Parameters whose `.grad` should be clipped (only those
            with a gradient set are considered - matches `Optimizer.step()`,
            which already skips parameters with `grad is None`).
        max_norm: The maximum allowed combined gradient norm.
        eps: Added to the denominator to avoid division by zero when the
            combined norm is (numerically) zero.

    Returns:
        The combined gradient norm *before* clipping (so callers can log
        it - a norm that's frequently near or above max_norm is a useful
        training-stability signal on its own).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    xp = get_array_module(grads[0])
    total_norm = xp.sqrt(sum(xp.sum(g * g) for g in grads))

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * clip_coef

    return float(total_norm)
