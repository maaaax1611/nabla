# Embedding

## The idea — not to be confused with "embeddings" the output

Two different things share the name "embedding," worth separating up
front:

- **The output of an encoder** — a dense vector representing a whole input
  (word, sentence, image), produced by many layers of transformation. This
  is what "embedding" usually means in a foundation-model context, and
  it's the *result* of running a model, not a layer in one.
- **`nn.Embedding`** (this doc) — the very first layer of a language model,
  turning a discrete token ID (that was created by a tokenizer) into the dense vector that then *feeds into*
  the encoder. It's a learnable lookup table, nothing more: shape
  `(vocab_size, embed_dim)`, one trainable row per vocabulary entry.

## The math (or lack of it)

$$
\text{out} = \text{table}[\text{indices}]
$$

Forward is pure indexing — for every integer ID in `indices`, look up the
corresponding row of `table`. No matmul, no activation, nothing to derive.
That simplicity is exactly what makes the *backward* pass worth pausing on.

## Why the backward pass isn't just "write the gradient back"

If `indices` never repeated within a call, backward would be trivial:
scatter `grad_output`'s rows back to the corresponding rows of `table`.
But token IDs *do* repeat — the same word can appear twice in one sequence,
or the same token across different sequences in a batch — and each
occurrence independently pulled `table`'s row into the output, so each
occurrence must independently contribute to that row's gradient. Writing
naively:

```python
grad_table[indices] = grad_output  # WRONG: last occurrence silently wins
```

would let a later repeat of the same ID silently overwrite an earlier
contribution instead of adding to it. This is exactly the same scatter/
accumulate problem [`MaxPool2D`'s backward](03-pooling.md) solves with
`np.add.at` — routing possibly-overlapping contributions back to a smaller
array without one clobbering another — so `Embedding` uses the identical
tool:

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
    grad_table = np.zeros(self.table_shape)
    np.add.at(grad_table, self.indices, grad_output)

    grad_indices = np.zeros_like(self.indices, dtype=float)
    return grad_table, grad_indices
```

`np.add.at(grad_table, self.indices, grad_output)` accumulates
`grad_output`'s rows into `grad_table` at the positions given by
`self.indices`, correctly summing when an index repeats — unlike plain
fancy-index assignment (`grad_table[self.indices] = ...`), which the NumPy
docs explicitly call out as *not* accumulating for repeated indices.

## `indices` isn't differentiable

Same pattern as `targets` in
[`SoftmaxCrossEntropy`](05-softmax-cross-entropy.md#targets-isnt-differentiable):
`indices` is still passed through `Function.apply` as a `Tensor`, purely so
it participates in the graph machinery like any other input, but integer
token IDs have no meaningful gradient. `backward` returns an all-zero array
for it, and `nn/embedding.py` never sets `requires_grad=True` on the
indices it's called with, so `Tensor.backward()` discards that gradient via
the same `if parent.requires_grad:` check.

## Initialization is different from Linear/Conv2D

`nn/embedding.py` doesn't default to He or Xavier init — both derive their
scale from `fan_in`/`fan_out`, which assumes every output depends on *all*
of a layer's inputs (true for a matmul, false here: each output row of an
embedding lookup depends on exactly *one* row of `table`, never a
combination of several). Instead it defaults to small-normal
initialization (`N(0, 0.01)`), matching PyTorch's `nn.Embedding` default:

```python
def _small_normal(shape: tuple[int, ...]) -> NDArray:
    return np.random.randn(*shape) * 0.01
```

## Testing

[`tests/test_embedding_ops.py`](../tests/test_embedding_ops.py) checks
forward against plain NumPy fancy indexing, backward against a numerical
gradient, and — the property that actually exercises the scatter-add fix —
that looking up the same index multiple times in one call produces a
gradient that's the *sum* of every occurrence's contribution, not just the
last one.
