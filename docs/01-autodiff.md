# Autodiff basics

## The idea

nabla builds a computation graph *as the program runs* (a "define-by-run" or
"tape-based" autodiff, the same style PyTorch uses) — as opposed to first
building a static graph and then executing it, like early TensorFlow did.
Every time you combine `Tensor`s with an operation, that operation records
itself as a node connecting its inputs to its output. Once you have a scalar
loss at the end of the graph, `backward()` walks the graph in reverse and
applies the chain rule at every node.

Two classes carry the whole design:

- **`Tensor`** ([`tensor.py`](../src/nabla/tensor.py)) wraps a NumPy array
  and, if `requires_grad=True`, remembers which operation produced it
  (`_ctx`) and what its inputs were (`_prev`).
- **`Function`** ([`function.py`](../src/nabla/function.py)) is the base
  class for every differentiable operation. Each op implements `forward`
  (compute the output, remember anything backward will need) and `backward`
  (given the gradient of the loss w.r.t. the output, return the gradient
  w.r.t. each input).

## Building the graph

`Function.apply` is the one place where a `Tensor` gets wired into the graph:

```python
@classmethod
def apply(cls, *inputs: Tensor, **kwargs) -> Tensor:
    ctx = cls(**kwargs)
    result = ctx.forward(*inputs)
    out = Tensor(result)

    out._ctx = ctx
    out._prev = inputs
    if any(t.requires_grad for t in inputs):
        out.requires_grad = True
    return out
```

So `x.relu()` (which calls `ReLU.apply(x)`) produces a *new* tensor whose
`_ctx` is the `ReLU` instance that computed it, and whose `_prev` is
`(x,)`. Chain a few ops together and you get a graph like:

```
x ──Linear──▶ h ──ReLU──▶ a ──Linear──▶ logits ──CrossEntropy──▶ loss
```

where every arrow is a `Function` instance sitting in the `_ctx` of the
tensor it points to.

## The chain rule, as a topological sort

For a scalar loss $L$ and any intermediate value $y$ that feeds into it,
the chain rule says:

$$
\frac{\partial L}{\partial y} = \sum_{z \text{ uses } y} \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y}
$$

In other words: to know $\partial L/\partial y$, you first need
$\partial L/\partial z$ for *every* $z$ that used $y$ as an input. That's
exactly what a **reverse topological order** guarantees — by the time you
process a node, every node that consumed its output has already been
processed. `Tensor.backward()` builds that order with a depth-first search:

```python
def topo_sort(tensor: Tensor) -> None:
    if id(tensor) not in visited:
        visited.add(id(tensor))
        for parent in tensor._prev:
            topo_sort(parent)
        topo.append(tensor)

topo_sort(self)
```

and then walks it back-to-front, calling each node's `Function.backward` and
accumulating (`+=`, never `=`) the result onto its parents' `.grad`:

```python
for tensor in reversed(topo):
    if tensor._ctx:
        grads = tensor._ctx.backward(tensor.grad)
        for parent, g in zip(tensor._prev, grads):
            if parent.requires_grad:
                parent.grad = g if parent.grad is None else parent.grad + g
```

Accumulation matters whenever a tensor is used more than once — e.g. a
weight shared across two branches of the graph gets gradient contributions
from both, and they need to *add up*, not overwrite each other.

## Broadcasting gradients back down (`unbroadcast`)

NumPy broadcasting means a forward op like `x + bias` (with `bias` shape
`(C,)` broadcasting against `x` shape `(N, C)`) can happily accept
mismatched shapes. The backward pass has to undo that: the incoming
gradient has the *broadcast* (output) shape, but `bias.grad` must end up in
`bias`'s own shape — by **summing** over every dimension that was
broadcast (since broadcasting is implicit tiling, and the gradient of a
sum of copies is the sum of their gradients).

[`broadcast.py`](../src/nabla/broadcast.py) handles this:

```python
def unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

> **Bug we hit:** the first version of this function only collapsed *extra
> leading* dimensions (the `while` loop) — the case where `bias` has fewer
> dimensions than `x`, like `(C,)` vs `(N, C)`. It missed the case where both
> shapes have the *same* number of dimensions but a size-1 axis got
> broadcast up, like `(1, C, 1, 1)` vs `(N, C, H, W)` — which is exactly
> what [BatchNorm2D](04-batchnorm.md) does in eval mode. Gradients came back
> in the wrong shape and crashed the next `Function.backward` downstream.
> The `for` loop above (summing any axis where `target_dim == 1` but
> `grad_dim != 1`) fixes the general case.

This is used by every elementwise op with two inputs
([`ops/basic.py`](../src/nabla/ops/basic.py): `Add`, `Subtract`, `Multiply`,
`Divide`), so the fix benefits all of them, not just BatchNorm.
