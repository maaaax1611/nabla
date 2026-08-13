# Scaled Dot-Product Attention

## The idea

Attention lets a model decide, per query position, how much to weigh each
of several key/value pairs — instead of every position in a sequence
processing itself in isolation (like a `Linear` layer would), attention
lets a query position pull in information from wherever in the sequence is
relevant to it. That's the operation multi-head attention (and Transformers
generally) build on:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $Q$ (queries), $K$ (keys), $V$ (values) are all just matrices — for a
  single sequence, $Q$ has one row per query position, $K$/$V$ have one row
  per key/value position (often the same sequence as $Q$, for
  *self*-attention).
- $QK^\top$ computes a dot-product similarity between every query and every
  key — row $i$, column $j$ is "how much does query $i$ relate to key $j$".
- Dividing by $\sqrt{d_k}$ (the query/key feature dimension) keeps those dot
  products from growing large as $d_k$ grows, which would otherwise push
  softmax into a saturated regime with near-zero gradients almost
  everywhere except the single largest score.
- `softmax` (row-wise) turns those similarity scores into a proper
  probability distribution per query — the *attention weights*.
- Multiplying by $V$ turns those weights into a weighted average of the
  value vectors — each query position's output is a blend of the values it
  attends to.

## Why this isn't its own `Function`

Every other op documented so far ([Conv2D](02-conv2d.md),
[Pooling](03-pooling.md), [BatchNorm2D](04-batchnorm.md)) is its own
`Function` with a hand-derived `backward`, because each one needed a
genuine reformulation trick (im2col, argmax-routing, the batch-statistics
chain rule) that doesn't fall out of composing existing ops. Attention is
different: $QK^\top$, the scale, the mask, the softmax, and the final
multiply by $V$ are *all* already differentiable — nothing here is new
math. So [`functional.scaled_dot_product_attention`](../src/nabla/functional.py)
is just those five steps written out as ordinary `Tensor` expressions, and
autodiff assembles the correct backward pass automatically by walking the
graph they build:

```python
scores = matmul(Q, _swap_last_two_axes(K))
scores = scores * Tensor(np.array(1 / np.sqrt(d_k)))
if mask is not None:
    scores = scores + Tensor(mask)

weights = softmax(scores, axis=-1)
return matmul(weights, V)
```

This is exactly why the prerequisite step was making
[`MatMul`](../src/nabla/ops/transform.py) support batched matmul (`np.matmul`
+ `np.swapaxes(-1, -2)` instead of `np.dot` + `.T`, with `broadcast.unbroadcast`
reducing any batch axes that got broadcast — the same helper `Add`/`Multiply`
already used for elementwise broadcasting) rather than adding a whole new
`Function` for attention — once the primitive is right, attention (and
later, multi-head attention) is just composition, not new calculus.

## Why it lives in `functional.py`, not as a `Tensor` method

Every argument here — $Q$, $K$, $V$ — is equally important; there's no
natural choice of which one should be `self` for a method call like
`Q.attention(K, V)`. That's exactly the case
[`functional.py`](../src/nabla/functional.py) exists for: a free function
`scaled_dot_product_attention(Q, K, V, mask=None)`, the same shape as
PyTorch's `torch.nn.functional` module and free functions like
`torch.matmul`. See the module's own docstring for the full reasoning.

## Shapes and batching

$$
Q: (*, L_q, d_k) \qquad K: (*, L_k, d_k) \qquad V: (*, L_k, d_v) \qquad (L = \text{seq len})
$$

The leading `*` axes are arbitrary — a plain batch axis today, and (once
multi-head attention splits `embed_dim` into per-head slices) a `num_heads`
axis too, with no code changes needed here. That's what
`_swap_last_two_axes` is for: it transposes only the trailing two axes of
`K` to get $K^\top$, no matter how many leading axes come before them —

```python
def _swap_last_two_axes(t: Tensor) -> Tensor:
    ndim = t.data.ndim
    axes = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return transpose(t, axes)
```

— which only works at all because `MatMul` now batches over those leading
axes the same way `np.matmul` does.

## The mask

`mask` is a plain `ndarray`, not a `Tensor` — there's nothing meaningful to
differentiate with respect to a fixed masking pattern. It's added *before*
the softmax, conventionally with a large negative value (e.g. `-1e9`) at
positions that must not be attended to: after softmax, `exp(-1e9)` underflows
to `0`, so those positions get (as close as floating point allows) exactly
zero attention weight without needing a separate masked-softmax
implementation.

## Testing

Because this is pure composition, correctness mostly means checking that
the composition is *assembled* correctly, not re-deriving any gradient by
hand. [`tests/test_functional.py`](../tests/test_functional.py) checks
forward against a naive reference (including the masked case), and backward
against a numerical gradient for all three inputs — the same numerical
gradient check used everywhere else in the library, just now confirming
that a whole chain of already-tested ops still composes into a correct
gradient end to end.
