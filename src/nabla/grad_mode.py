from __future__ import annotations

_grad_enabled = True


def is_grad_enabled() -> bool:
    """Whether operations should currently attach themselves to a graph."""
    return _grad_enabled


class no_grad:
    """Context manager that disables graph construction for its block.

    Every `Function.apply` call normally attaches its Function instance
    (and everything it saved for backward - e.g. Conv2D's full im2col
    buffer) to the output Tensor via `_ctx`/`_prev`, keeping the whole
    forward pass's intermediates alive for as long as that output Tensor
    is reachable, whether or not `.backward()` is ever actually called.
    A validation/inference pass never calls backward, so without this
    there is no way to avoid paying that memory cost anyway - only
    reassigning the output variable eventually frees it, by which point a
    training step may already need that memory back.

    Usage:
        with no_grad():
            val_loss = compute_loss(model, X_val, y_val, criterion, device)
    """

    def __enter__(self) -> "no_grad":
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc_info: object) -> bool:
        global _grad_enabled
        _grad_enabled = self._previous
        return False
