---
title: "Autograd from scratch"
description: "Tensors, Functions, the computational graph, and how backward() turns the chain rule into a topological sort."
---

# Autograd from scratch

This is the first post in a series that walks through **nabla**, a small
automatic differentiation library built on NumPy. Everything later in the
series — convolutions, attention, a Vision Transformer — rests on the two
classes introduced here and on one idea: record what happens during the
forward pass, then replay it backwards with the chain rule.

By the end of this post you should be able to read
[`tensor.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/tensor.py) and
[`function.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/function.py) top to bottom and know why each
line is there.

## Autodiff basics


nabla builds a computation graph *as the program runs* (a "define-by-run" or
"tape-based" autodiff, the same style PyTorch uses) — as opposed to first
building a static graph and then executing it, like early TensorFlow did.
Every time you combine `Tensor`s with an operation, that operation records
itself as a node connecting its inputs to its output. Once you have a scalar
loss at the end of the graph, `backward()` walks the graph in reverse and
applies the chain rule at every node.

Two classes carry the whole design:

- **`Tensor`** ([`tensor.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/tensor.py)) wraps a NumPy array
  and, if `requires_grad=True`, remembers which operation produced it
  (`_ctx`) and what its inputs were (`_prev`).
- **`Function`** ([`function.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/function.py)) is the base
  class for every differentiable operation. Each op implements `forward`
  (compute the output, remember anything backward will need) and `backward`
  (given the gradient of the loss w.r.t. the output, return the gradient
  w.r.t. each input).

### Building the graph

`Function.apply` is the one place where a `Tensor` gets wired into the graph:

```python
@classmethod
def apply(cls, *inputs: Tensor, **kwargs) -> Tensor:
    ctx = cls(**kwargs)
    result = ctx.forward(*inputs)
    out = Tensor(result)

    if _grad_enabled and any(t.requires_grad for t in inputs):
        out._ctx = ctx
        out._prev = inputs
        out.requires_grad = True
    return out
```

The `if` is what keeps the graph as small as it can be: a result is only
wired in when a gradient could actually flow through it. An op on plain
constants produces a plain tensor, and its `Function` — together with
everything it saved for the backward pass — is dropped on the spot.
(`_grad_enabled` is the switch behind [`no_grad`](#turning-the-tape-off-no_grad),
further down.)

So `x.relu()` (which calls `ReLU.apply(x)`) produces a *new* tensor whose
`_ctx` is the `ReLU` instance that computed it, and whose `_prev` is
`(x,)`. Chain a few ops together and you get a graph like:

```
x ──Linear──▶ h ──ReLU──▶ a ──Linear──▶ logits ──CrossEntropy──▶ loss
```

where every arrow is a `Function` instance sitting in the `_ctx` of the
tensor it points to.

### The chain rule, as a topological sort

For a scalar loss $L$ and any intermediate value $y$ that feeds into it,
the chain rule says:

$$
\frac{\partial L}{\partial y} = \sum_{z \text{ uses } y} \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y}
$$

In other words: to know $\partial L/\partial y$, you first need
$\partial L/\partial z$ for *every* $z$ that used $y$ as an input. That's
exactly what a **reverse topological order** guarantees — by the time you
process a node, every node that consumed its output has already been
processed. `Tensor.backward()` builds that order with a depth-first search. In its
simplest, recursive form:

```python
def topo_sort(tensor: Tensor) -> None:
    if id(tensor) not in visited:
        visited.add(id(tensor))
        for parent in tensor._prev:
            topo_sort(parent)
        topo.append(tensor)

topo_sort(self)
```

(The version in the repository does the same traversal with an explicit
stack instead of recursion. The reason is not style — the recursive closure
turned out to be one half of a memory leak, which is the story
[at the end of this post](#graph-memory-and-gpu-stalls).)

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

### Broadcasting gradients back down (`unbroadcast`)

NumPy broadcasting means a forward op like `x + bias` (with `bias` shape
`(C,)` broadcasting against `x` shape `(N, C)`) can happily accept
mismatched shapes. The backward pass has to undo that: the incoming
gradient has the *broadcast* (output) shape, but `bias.grad` must end up in
`bias`'s own shape — by **summing** over every dimension that was
broadcast (since broadcasting is implicit tiling, and the gradient of a
sum of copies is the sum of their gradients).

[`broadcast.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/broadcast.py) handles this:

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
> what [BatchNorm2D](04-cnn.md#batchnorm2d) does in eval mode. Gradients came back
> in the wrong shape and crashed the next `Function.backward` downstream.
> The `for` loop above (summing any axis where `target_dim == 1` but
> `grad_dim != 1`) fixes the general case.

This is used by every elementwise op with two inputs
([`ops/basic.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/basic.py): `Add`, `Subtract`, `Multiply`,
`Divide`), so the fix benefits all of them, not just BatchNorm.

## Small primitives, composed

A `Function` is worth writing by hand in two situations: the op is a true
primitive (there is nothing simpler to build it from), or a fused version is
numerically safer or much cheaper than the composition. Everything else can
be assembled from ops that already know their own derivative — autodiff
then produces the backward pass for free.

The elementwise primitives live in
[`ops/basic.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/basic.py) and
[`ops/activation.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/activation.py). Each backward is one
line of calculus:

| Op | Forward | Local derivative |
| --- | --- | --- |
| `Neg` | $-x$ | $-1$ |
| `Exp` | $e^x$ | $e^x$ (the forward result, reused) |
| `Log` | $\ln x$ | $1/x$ |
| `Pow` | $x^p$, constant $p$ | $p\,x^{p-1}$ |
| `Tanh` | $\tanh x$ | $1 - \tanh^2 x$ |

`Exp` is the smallest example of an op saving its *output* rather than its
input for the backward pass:

```python
class Exp(Function):
    def forward(self, x):
        xp = get_array_module(x.data)
        self.out = xp.exp(x.data)
        return self.out

    def backward(self, grad_output):
        return (grad_output * self.out,)
```

`Pow` takes its exponent as a plain Python number, not a `Tensor`. It is
configuration, like `axis` on a softmax — so `x ** 2`, `x ** 0.5`
(`x.sqrt()`) and `x ** -1` all work, and there is no gradient with respect
to the exponent to worry about.

### Composition in action: softmax without a `Softmax` op

With `exp`, a sum over one axis and a division, softmax is three lines:

```python
e = x.exp()
probs = e / e.sum(axis=-1, keepdims=True)
```

Nobody derived a gradient for this. `Exp`, `Sum` and `Divide` each know
their own local derivative, and `backward()` chains them. The test suite
checks that the result agrees with the hand-derived `Softmax` op from the
[next post](02-mlp.md#softmax) to floating-point precision
([`tests/test_elementwise_math_ops.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_elementwise_math_ops.py)).

So why does nabla still ship a fused `Softmax`? Because the composed version
overflows as soon as a logit gets large — $e^{1000}$ is `inf` — and the
fused op can subtract the row maximum first. Composition gives you
correctness for free; fusion buys numerical stability and speed. The same
trade-off shows up again for softmax + cross-entropy, LayerNorm and
BatchNorm.

### Reductions over an axis

`sum()` and `mean()` reduce everything to a scalar by default, and take
`axis` / `keepdims` like their NumPy counterparts
([`ops/reduce.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/reduce.py)).

The backward pass of a sum is the mirror image of broadcasting. Every input
element contributed to exactly one output element with weight 1, so each
input position receives the gradient of the output cell it was summed into:

```python
def _expand_reduced(grad, shape, axis, keepdims):
    if axis is not None and not keepdims:
        for ax in sorted(a % len(shape) for a in axes):
            grad = xp.expand_dims(grad, ax)   # put the reduced axes back
    return xp.broadcast_to(grad, shape)       # copy along them
```

`Mean` is the same with weight $1/n$, where $n$ is the number of elements
averaged into one output cell. Note the symmetry with `unbroadcast` above:
broadcasting copies forward and sums backward; a reduction sums forward and
copies backward.

> **Detail worth knowing:** `broadcast_to` returns a *read-only view*. `Sum`
> returns a copy of it, because optimizers and gradient clipping modify
> `.grad` in place — a read-only gradient would crash far away from the op
> that produced it.

### Constants as operands

Writing `x * Tensor(np.array(0.5))` for every constant gets old quickly, so
the arithmetic operators accept Python numbers and plain arrays and wrap
them on the fly:

```python
def _wrap(self, other):
    if isinstance(other, Tensor):
        return other
    xp = get_array_module(self.data)
    dtype = self.data.dtype if self.data.dtype.kind == "f" else None
    return Tensor(xp.asarray(other, dtype=dtype))
```

Two details in those few lines matter more than they look:

- The constant is created with **this tensor's array module**, so it lands
  on the GPU when the tensor lives there (see the
  [engineering post](07-engineering.md#gpu-support-via-cupy)).
- It takes **this tensor's dtype**. A zero-dimensional `float64` array is not
  a "weak" scalar in NumPy's promotion rules — multiplying a `float32` array
  by one silently upgrades the whole result to `float64`.

The reflected operators (`__radd__`, `__rsub__`, `__rmul__`,
`__rtruediv__`) cover the case where the constant is on the left: `1 - x`
calls `x.__rsub__(1)`. For `ndarray - Tensor` NumPy would normally try to
handle the operation itself, elementwise over the tensor as an opaque
object; setting `__array_ufunc__ = None` on `Tensor` makes NumPy step aside
so Python falls through to `Tensor.__rsub__`.

## Turning the tape off: `no_grad`

Recording the graph is only useful if someone calls `backward()` later.
During evaluation or sampling nobody does, and every recorded op would keep
its inputs alive for nothing. `no_grad` switches recording off:

```python
with nabla.no_grad():
    logits = model(x)

@nabla.no_grad()
def accuracy(model, X, y): ...
```

It is a module-level flag in [`function.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/function.py) that
`Function.apply` checks — the same line that already skips recording when
no input requires a gradient. The context manager saves the previous value
on entry and restores it on exit, so blocks nest correctly and an exception
inside the block cannot leave gradients disabled.

## Graph Memory and GPU Stalls

`src/nabla/tensor.py` (`Tensor.backward()`)

This one didn't start as a memory-management investigation - it started as
"scale the Shakespeare model up 30x and see what happens." What happened
was a training loop that ran at wildly inconsistent speeds, from under a
second to over 40 seconds per step, with no code difference between steps.

### The stall: the symptom

Scaling `examples/shakespeare_transformer.py` from ~158K parameters
(embed_dim=64, 3 layers) to ~4.8M parameters (embed_dim=256, 6 layers,
longer sequences) made per-step time on the GPU completely erratic:

```
step 0: 806.5 ms   | pool used 4128.0 MB
step 1: 1438.0 ms  | pool used 8103.1 MB
step 2: 4150.5 ms  | pool used 12078.2 MB
step 3: 4525.5 ms  | pool used 16053.2 MB
step 4: 8409.0 ms  | pool used 4128.0 MB   <- drops back down
step 5: 2988.5 ms  | pool used 8103.1 MB
...
step 9: 13059.0 ms | pool used 4128.0 MB
```

CuPy's memory pool (`cupy.get_default_memory_pool().used_bytes()`) grew by
roughly one step's worth of activations *every single step*, for several
steps in a row, before suddenly dropping back down - and the steps where
it dropped were the slowest ones by far. The smaller model never showed
this pattern at all; its memory usage was flat step to step.

### The stall: chasing it down

The growing-then-dropping pattern is the signature of relying on Python's
*cyclic* garbage collector instead of plain reference counting. CPython
frees an object the instant its refcount hits zero - but if two objects
reference each other (directly or through a longer chain), their
refcounts never reach zero on their own, no matter how unreachable the
whole cycle is from the rest of the program. Only the generational
collector's cycle-finder can free those, and it doesn't run on every
allocation - it runs based on an internal allocation-count heuristic. So
memory tied up in cycles keeps accumulating until that collector happens
to fire, at which point everything gets freed in one lump - which is
exactly what step 4's 8.4-second stall was: CuPy's allocator scrambling to
service new allocations while GPU memory was still full of the previous
several steps' un-freed graphs, right up until gc finally ran mid-step.

Confirming it: manually calling `gc.collect()` after `backward()`
(instead of waiting on Python to decide) immediately dropped the pool from
4.1 GB to 153 MB and reported reclaiming hundreds of objects - proof the
"leaked" memory was cyclic garbage, not a real leak.

`gc.get_referrers()` on the collected objects pointed at two independent
cycles:

**1. The autodiff graph itself.** Every non-leaf `Tensor` holds a
`Function` via `_ctx`; every `Function` holds its input tensors via
`saved_tensors`. That's not a cycle by itself (children point at parents,
parents never point forward at children) - but once a full graph has been
built and walked, nothing needs those links anymore. `Tensor.backward()`
now clears `_ctx` and `_prev` on every tensor in the graph right after
computing gradients, the same "free the graph by default" behavior
PyTorch has when you don't pass `retain_graph=True`.

**2. `topo_sort`'s own closure.** The topological sort inside
`backward()` used to be a recursive nested function:

```python
def topo_sort(tensor: Tensor) -> None:
    if id(tensor) not in visited:
        visited.add(id(tensor))
        for parent in tensor._prev:
            topo_sort(parent)   # <- refers to itself
        topo.append(tensor)

topo_sort(self)
```

A nested function that calls itself by name resolves that name through
its own closure cell - which means the function object holds a reference
to a cell that (once the recursive call has happened) refers back to the
function object itself. That's a reference cycle completely independent
of anything tensor-related, and it kept `backward()`'s entire stack frame
- including the `topo` list, which by construction holds a reference to
*every tensor in the graph* - alive until the cyclic collector ran. This
one mattered more than it looks: even after fix #1 cleared every tensor's
own `_ctx`/`_prev`, the `topo` list still held direct references to all
of them, so the graph stayed alive regardless.

The fix is to make the traversal iterative instead - an explicit stack
carrying `(tensor, expanded)` pairs reproduces the same post-order a
recursive DFS would, with no closure and therefore no self-reference:

```python
topo: list[Tensor] = []
visited: set[int] = set()
stack: list[tuple[Tensor, bool]] = [(self, False)]
while stack:
    tensor, expanded = stack.pop()
    if expanded:
        topo.append(tensor)
    elif id(tensor) not in visited:
        visited.add(id(tensor))
        stack.append((tensor, True))
        for parent in tensor._prev:
            stack.append((parent, False))
```

### The stall: the result

With both cycles gone, the 4.8M-parameter model's per-step time on the
GPU became flat and predictable - no `gc.collect()` workaround needed in
the training script itself, because there's no cyclic garbage left for it
to have to clean up:

```
step 0: 910.0 ms | pool used 153.0 MB
step 1: 701.0 ms | pool used 153.0 MB
step 2: 700.0 ms | pool used 153.0 MB
...
step 14: 709.0 ms | pool used 153.0 MB
```

### Why the small model never showed this

Both cycles existed the whole time - they're not new bugs introduced by
scaling up. What changed is how much memory got trapped in each one
before Python's cyclic collector happened to run. At 158K parameters, a
few un-freed graphs are a rounding error next to an 8 GB GPU; the
allocator never had to work hard to satisfy a new request, so nothing
ever stalled long enough to notice. At 4.8M parameters (and longer
sequences, and a bigger batch), a handful of un-freed graphs was already
gigabytes, so the same collection-timing behavior turned into
multi-second, once-every-few-steps stalls. This is the same shape of
lesson as the [GPU dispatch bugs](07-engineering.md#gpu-support-via-cupy): correctness bugs
that a small enough test case simply won't reveal.

#### Testing

[`tests/test_graph_lifecycle.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_graph_lifecycle.py) checks
that `_ctx`/`_prev` are cleared on both intermediate and root tensors
after `backward()`, that leaf tensors are untouched and remain reusable
in a fresh graph afterward, that clearing the graph doesn't disturb the
gradients that were computed from it, and - with `gc` disabled so only
plain refcounting can act - that a `matmul` + `sum` graph leaves zero
objects for the cyclic collector to find at all.

---

Next: [From a neuron to an MLP](02-mlp.md)
