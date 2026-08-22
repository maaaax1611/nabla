# no_grad

## Overview

`nabla.grad_mode.no_grad` is a context manager that disables computation-
graph construction for the code inside its `with` block. Use it around
any forward pass that will never call `.backward()` - typically
validation/evaluation - to avoid paying the memory cost of a graph that
will never be used.

```python
with no_grad():
    val_loss = compute_loss(model, X_val, y_val, criterion, device)
```

## Math

None - this changes what `Function.apply` attaches to a `Tensor`, not
any forward or backward computation itself.

## Why it's needed

`Function.apply` (see [Autodiff basics](01-autodiff.md)) unconditionally
attaches its `Function` instance to the output `Tensor` via `_ctx`/
`_prev`, regardless of whether `.backward()` will ever be called on it.
That `Function` instance holds everything it saved for its own backward
pass - for `Conv2D`, that includes its full im2col buffer, which for a
large model at full resolution can be gigabytes per layer (see
[U-Net](28-unet.md)). As long as the output `Tensor` stays reachable
(e.g. a `val_loss` local variable that isn't reassigned until the next
evaluation), that entire chain of buffers stays alive in memory too -
useful for a training step (backward needs it), pure waste for a
validation step (backward is never called).

## Implementation

```python
_grad_enabled = True

def is_grad_enabled() -> bool:
    return _grad_enabled

class no_grad:
    def __enter__(self) -> "no_grad":
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc_info: object) -> bool:
        global _grad_enabled
        _grad_enabled = self._previous
        return False
```

`Function.apply` checks this flag before attaching anything:

```python
ctx = cls(**kwargs)
result = ctx.forward(*inputs)
out = Tensor(result)

if is_grad_enabled():
    out._ctx = ctx
    out._prev = inputs
    if any(t.requires_grad for t in inputs):
        out.requires_grad = True

return out
```

Inside a `no_grad()` block, `ctx` is a local variable inside `apply()`
that's never attached to anything the caller keeps - once `apply()`
returns, its reference count drops to zero and CPython frees it (and
whatever it saved) immediately via plain refcounting, no cyclic GC
involved (contrast with the reference-cycle issue in
[Graph Memory and GPU Stalls](25-graph-memory-and-gpu-stalls.md), which
was about cleaning up *after* a real backward pass - this sidesteps
building the graph at all).

A global flag (rather than, say, a per-Tensor flag) mirrors PyTorch's
`torch.no_grad()`: simple to reason about, and nesting works for free -
the previous value is saved and restored around each block, so a nested
`no_grad()` inside another just leaves the outer disabled state in place
on exit.

## Testing

[`tests/test_grad_mode.py`](../tests/test_grad_mode.py) checks the flag's
default state, that it's disabled inside the block and restored after
(including on an exception and when nested), that outputs built inside
the block have `requires_grad=False` and no `_ctx`/`_prev`, that forward
values themselves are unaffected, and that a graph built outside any
`no_grad()` block is unaffected.
