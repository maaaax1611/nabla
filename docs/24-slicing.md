# Slicing

`src/nabla/ops/slice.py`

Every op before this one either kept a tensor's full shape (elementwise ops,
`LayerNorm`) or transformed it in a way that's still a *function of every
input element* (`Reshape`, `Transpose`, `MatMul`, reductions). `Slice` is the
first op that reads out only *some* of a tensor's elements and throws the
rest away — `x[:, 0]`, `x[1:3]`, `x[..., :2]`. It's what `Tensor.__getitem__`
is built on.

## Why this was missing until now

Before `Slice` existed, `VisionTransformer` needed to pick out a single
sequence position (the CLS token) and had no way to do it, so
[docs/17](17-vision-transformer.md) worked around the gap with a fixed
one-hot row and a `MatMul`:

```python
selector = np.zeros((1, 1, num_patches + 1))
selector[0, 0, 0] = 1.0
cls_out = F.matmul(selector, x).reshape((batch, -1))
```

That's correct — multiplying by a one-hot row *is* mathematically the same
as indexing — but it's a workaround, not the operation anyone would reach
for first. With `Slice` in place, `VisionTransformer.forward` now just does:

```python
cls_out = x[:, 0]  # (batch, embed_dim)
```

## Forward

`Slice` wraps ordinary numpy *basic indexing* — an int, a `slice`, `Ellipsis`,
`None` (newaxis), or a tuple of those. It deliberately does **not** support
boolean masks or integer-array ("fancy") indexing:

```python
def forward(self, x: Tensor) -> NDArray:
    self.original_shape = x.data.shape
    return x.data[self.key]
```

The only thing `backward` needs back is the input's original shape — `key`
itself is already stored on `self` from `__init__`, exactly like `Transpose`
stores `self.axes`.

## Backward

Every output element of a basic-indexing slice traces back to *exactly one*
input element — there's no overlap the way `MaxPool2D` or `Embedding` can
have (an embedding table row can be looked up by two different token IDs in
the same batch; a slice never reads the same source element twice). So the
backward pass is a plain scatter, not a scatter-*accumulate*:

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray]:
    xp = get_array_module(grad_output)
    grad_x = xp.zeros(self.original_shape, dtype=grad_output.dtype)
    grad_x[self.key] = grad_output
    return (grad_x,)
```

Build a zero array shaped like the original input, and write `grad_output`
into it at the same `key` that picked it out during `forward` — everywhere
that wasn't read gets a zero gradient, since it had no effect on the output.

This is a genuinely different shape of problem than `Embedding.backward`,
which needs `xp.add.at` because indices *can* repeat:

```python
# Embedding.backward - indices can repeat, so contributions must sum
xp.add.at(grad_table, self.indices, grad_output)

# Slice.backward - basic indexing never repeats a source element
grad_x[self.key] = grad_output
```

Using assignment here is not a shortcut that happens to work for the test
cases at hand — it's the correct operation for exactly this class of
indexing. If `Slice` ever grows to support fancy (integer-array) indexing,
`x[[0, 0, 1]]` would repeat index `0`, and backward would need to switch to
`xp.add.at`, same as `Embedding`.

One subtlety worth calling out explicitly: `Slice.backward` itself never
needs to worry about *two separate slices* overlapping — e.g. `x[0:2]` and
`x[1:3]` on the same tensor both touching index `1`. That's handled one
level up, in `Tensor.backward()`'s gradient accumulation (`parent.grad =
parent.grad + g`), the same mechanism that lets any tensor feed into
multiple downstream ops. Each `Slice` only ever scatters its own
`grad_output` into its own zero array; summing contributions across
multiple ops touching the same tensor was already `Tensor.backward()`'s job
before `Slice` existed.

## Tests

`tests/test_slice_ops.py` covers: forward with an int index, a slice, a
tuple key that drops an axis (the actual ViT CLS pattern), indexing via
`x[...]` syntax, backward against a hand-computed expected gradient,
backward against central-difference numerical gradients, `Ellipsis` +
`None` in the key, and two overlapping slices of the same tensor summing
their gradients correctly at the overlap (exercising the
`Tensor.backward()` accumulation path described above, not `Slice` itself).
