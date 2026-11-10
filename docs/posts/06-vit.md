---
title: "Vision Transformer"
description: "Patches as tokens, a CLS token via Concat, slicing, and how a ViT compares to a CNN on MNIST."
---

# Vision Transformer

The [Transformer](05-transformer.md) from the previous post operates on a
sequence of vectors and does not care where they come from. A Vision
Transformer cuts an image into patches, treats each patch as a token and
reuses the same blocks unchanged. Getting there needed two small ops —
concatenation and slicing — which are covered here along with the model
itself and how it compares to the [CNN](04-cnn.md) on MNIST.

## Concat

### Why we need it

Every op so far takes a fixed, small number of tensors. The
[Vision Transformer](#vision-transformer-vit) needs something new: a
learnable **CLS token** prepended to the sequence of patch embeddings
before the Transformer stack runs over it — the same trick BERT and ViT
both use, a dummy "summary" position whose final output feeds the
classification head. That's concatenation along the sequence axis, and
nothing in nabla could do that yet.

### Concat: the op

[`ops/concat.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/concat.py)'s `Concat` is the first op
in nabla that takes a *variable* number of tensors rather than one or
two — `Function.apply(*inputs, **kwargs)` and
`Function.backward(...) -> tuple[...]` already supported this, they just
hadn't been exercised by an op that needed it yet.

```python
def forward(self, *tensors):
    self.save_for_backward(*tensors)
    return np.concatenate([t.data for t in tensors], axis=self.axis)
```

### Concat: the backward pass

Concat glues its inputs together along one axis with no overlap — every
element of the output came from exactly *one* input, at exactly the
position it already had. That makes the backward pass the simplest kind
there is: no summing, no reduction, just cutting `grad_output` back into
the same pieces it was assembled from, using each input's original size
along that axis:

```python
def backward(self, grad_output):
    tensor_sizes = [t.data.shape[self.axis] for t in self.saved_tensors]
    split_indices = np.cumsum(tensor_sizes)[:-1]
    return tuple(np.split(grad_output, split_indices, axis=self.axis))
```

`np.split` wants the *cut points* between pieces, not each piece's size —
for sizes `[3, 5, 2]` that's `cumsum([3, 5, 2])[:-1] == [3, 8]` (drop the
final cumulative total, since `np.split` doesn't need a trailing cut at
the very end of the array).

### The off-by-one bug that came up

An earlier version computed `split_indices` as
`cumsum([0] + tensor_sizes)[:-1]`. Prepending that `0` shifts every cut
point down by one slot and, worse, adds a phantom leading split: for
three tensors, `np.split` returns *four* pieces (`[0:0]`, `[0:3]`,
`[3:8]`, `[8:10]`) — an empty array first, then everything after it
misaligned with the wrong input. Numerical gradient checking on all
three inputs caught it immediately. The fix was just to drop the
leading `0` — `cumsum(tensor_sizes)[:-1]` already starts at the first
real cut point.

#### Testing

[`tests/test_concat_ops.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_concat_ops.py) checks forward
shapes and values for two and three tensors, that `backward` routes each
slice of `grad_output` back to exactly the right input (checked both by
direct slicing and by central-difference numerical gradient checking),
and that gradient shapes always match their input's shape even when
inputs have very different sizes along the concatenation axis.

## Vision Transformer (ViT)


A Transformer operates on a sequence of vectors — it has no idea whether
those vectors came from word embeddings or something else entirely. ViT
(Dosovitskiy et al., "An Image is Worth 16x16 Words") exploits exactly
that: cut an image into fixed-size patches, linearly project each patch
to a vector, and hand the resulting sequence to the *same* Transformer
stack already built for [text](05-transformer.md#example-a-character-level-transformer-on-tiny-shakespeare). Every op this
needed either already existed or was pure composition — no new attention
mechanism, no vision-specific math.

### Patches via Conv2D, not a new op

[`nn/patch_embedding.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/patch_embedding.py)'s
`PatchEmbedding` turns an image into a sequence with a single
[`Conv2D`](04-cnn.md#conv2d) where `kernel_size == stride == patch_size`:

```python
self.proj = Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

Each convolution window then covers exactly one patch with zero overlap
(stride equals kernel size), and the kernel's `out_channels` axis *is*
the per-patch linear projection — mathematically identical to flattening
each patch into a vector and running it through a `Linear` layer, just
without ever materializing the flattened patches. `Conv2D`'s backward
already exists; nothing new to derive. The `(batch, embed_dim, grid,
grid)` output gets reshaped and transposed into a `(batch, num_patches,
embed_dim)` sequence — the shape every Transformer block expects.

### The CLS token, and why `Concat` was worth building for it

Following BERT/ViT convention, a single learnable **CLS token** is
prepended to the patch sequence. After the Transformer stack runs, *its*
final representation (not any patch's) feeds the classification head —
a dedicated "summary" position that's free to attend to (and be attended
by) every patch, rather than forcing one arbitrary patch to double as
the whole image's summary. Prepending a fixed-size learnable vector to a
per-batch sequence is exactly what motivated
[`Concat`](#concat):

```python
cls_tokens = self.cls_token + Tensor(np.zeros((batch, 1, embed_dim)))
x = F.concat([cls_tokens, x], axis=1)
```

(the `+ zeros` broadcasts the single learned CLS vector to every item in
the batch before concatenation, since `Concat` itself doesn't broadcast —
every non-concatenated axis has to already match.)

### Learned positional embedding, not sinusoidal

Unlike the [Shakespeare example](05-transformer.md#example-a-character-level-transformer-on-tiny-shakespeare)'s fixed
sinusoidal table, ViT uses a **learned** positional embedding — a plain
`Tensor(requires_grad=True)` of shape `(1, num_patches + 1, embed_dim)`,
added once after the CLS token is prepended:

```python
self.pos_embedding = Tensor(np.random.randn(1, num_patches + 1, embed_dim) * 0.02, requires_grad=True)
...
x = x + self.pos_embedding
```

No new op needed — `Add` already broadcasts a `(1, seq, dim)` embedding
across the batch axis and its backward already sums the broadcast
gradient back down (the exact mechanism [Attention](05-transformer.md#multi-head-attention)'s
batched `MatMul` relies on too). Patches have no inherent left-to-right
order the way text tokens do, and unlike text there's no need to
generalize to unseen sequence lengths at inference time (image size,
hence patch count, is fixed per model) — so a learned table is the
simpler and empirically stronger choice for ViT specifically.

### No mask

`TransformerBlock.forward(x, mask=None)` runs with no mask at all here —
unlike causal language modeling, an image classifier has no
"future" to hide from a patch. Every patch (and the CLS token) may
freely attend to every other patch.

### Picking out the CLS token

After the Transformer stack and a final `LayerNorm`, only the CLS
token's output (position 0) goes to the classification head:

```python
cls_out = x[:, 0]  # (batch, embed_dim)
```

**Originally**, nabla had no `Tensor.__getitem__`/slicing op, so this used
a fixed (non-learnable, `requires_grad=False`) one-hot row and `MatMul`
instead — `selector @ x` sums over the sequence axis with weight 1 at
position 0 and 0 everywhere else, which is exactly `x[:, 0, :]`. That
workaround is gone now that [`Slice`](#slicing) exists; see that doc
for the backward-pass derivation (it's a plain scatter, not the
scatter-*accumulate* `Embedding` needs, because basic indexing never reads
the same source element twice).

#### Testing

[`tests/test_nn_patch_embedding.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_patch_embedding.py)
checks patch-count/shape math, rejects non-divisible patch sizes, and
verifies patches are spatially independent (zeroing one patch's pixels
must not change any other patch's embedding). [`tests/test_nn_vit.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_vit.py)
checks end-to-end forward shape (including a multi-channel/RGB case),
that gradients reach every parameter including `cls_token` and
`pos_embedding`, and that `train()`/`eval()` propagates down into the
block stack's dropout layers.

## Slicing

`src/nabla/ops/slice.py`

Every op before this one either kept a tensor's full shape (elementwise ops,
`LayerNorm`) or transformed it in a way that's still a *function of every
input element* (`Reshape`, `Transpose`, `MatMul`, reductions). `Slice` is the
first op that reads out only *some* of a tensor's elements and throws the
rest away — `x[:, 0]`, `x[1:3]`, `x[..., :2]`. It's what `Tensor.__getitem__`
is built on.

### Why this was missing until now

Before `Slice` existed, `VisionTransformer` needed to pick out a single
sequence position (the CLS token) and had no way to do it, so
the [model above](#vision-transformer-vit) worked around the gap with a fixed
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

### Slicing: forward

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

### Slicing: backward

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

#### Testing

`tests/test_slice_ops.py` covers: forward with an int index, a slice, a
tuple key that drops an axis (the actual ViT CLS pattern), indexing via
`x[...]` syntax, backward against a hand-computed expected gradient,
backward against central-difference numerical gradients, `Ellipsis` +
`None` in the key, and two overlapping slices of the same tensor summing
their gradients correctly at the overlap (exercising the
`Tensor.backward()` accumulation path described above, not `Slice` itself).

## Example: MNIST Digit Classification with a Vision Transformer

### ViT on MNIST: what this ties together

[`examples/mnist_vit.py`](https://github.com/maaaax1611/nabla/blob/main/examples/mnist_vit.py) trains
[`VisionTransformer`](#vision-transformer-vit) on MNIST digit
classification, reusing the same [`DataLoader`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/data/dataloader.py)
and MNIST loading code as
[`examples/mnist_cnn.py`](https://github.com/maaaax1611/nabla/blob/main/examples/mnist_cnn.py) — only the model and
training loop change.

```python
model = VisionTransformer(
    img_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, num_heads=4, hidden_dim=128, num_layers=4, dropout=0.1,
)
```

`28 / 4 = 7`, so each 28x28 digit becomes a 7x7 grid — 49 patches, plus
the CLS token, for a sequence length of 50 fed through 4 stacked
`TransformerBlock`s.

### ViT on MNIST: training loop

Identical in shape to `mnist_cnn.py`'s: `CrossEntropyLoss`, `Adam`,
`zero_grad()` / `backward()` / `step()` per batch, `train()`/`eval()`
toggled around each epoch's evaluation pass. Nothing ViT-specific here —
once the model itself is built, "train an image classifier" looks the
same regardless of what's inside the model.

### ViT on MNIST: results

10 epochs over a 3000-image training subset (same budget as
`mnist_cnn.py`), on CPU:

```
Epoch  1 | Train loss: 1.7570 | Test accuracy: 58.20%
Epoch  5 | Train loss: 0.7496 | Test accuracy: 76.80%
Epoch 10 | Train loss: 0.5425 | Test accuracy: 81.60%
```

~2.5-3 minutes total. Don't expect this to beat `mnist_cnn.py`'s
accuracy at a comparable budget — a CNN has translation-equivariance
built into its architecture (a learned filter that recognizes an edge
works the same wherever the edge appears), which a Transformer has to
learn from data instead. This gap is a well-known, real property of
ViT, not a nabla-specific issue: the original ViT paper needed
much larger datasets (or pretraining) to match or beat CNNs. The point
of this example is exercising every ViT building block end to end —
`Conv2D`-as-patch-projection, `Concat`-prepended CLS token, learned
positional embeddings, a full Transformer stack, `MatMul`-based
token selection — not chasing state-of-the-art accuracy.

### ViT on MNIST: running it

```bash
uv run python examples/mnist_vit.py
```

MNIST downloads once and is cached under `examples/.mnist_cache/`
(git-ignored, shared with `mnist_cnn.py`) on first run.

---

Previous: [Attention and the Transformer](05-transformer.md) · Next: [Engineering: GPU, precision, checkpoints, logging](07-engineering.md)
