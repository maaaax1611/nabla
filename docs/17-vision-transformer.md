# Vision Transformer (ViT)

## The core idea

A Transformer operates on a sequence of vectors — it has no idea whether
those vectors came from word embeddings or something else entirely. ViT
(Dosovitskiy et al., "An Image is Worth 16x16 Words") exploits exactly
that: cut an image into fixed-size patches, linearly project each patch
to a vector, and hand the resulting sequence to the *same* Transformer
stack already built for [text](15-shakespeare-example.md). Every op this
needed either already existed or was pure composition — no new attention
mechanism, no vision-specific math.

## Patches via Conv2D, not a new op

[`nn/patch_embedding.py`](../src/nabla/nn/patch_embedding.py)'s
`PatchEmbedding` turns an image into a sequence with a single
[`Conv2D`](02-conv2d.md) where `kernel_size == stride == patch_size`:

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

## The CLS token, and why `Concat` was worth building for it

Following BERT/ViT convention, a single learnable **CLS token** is
prepended to the patch sequence. After the Transformer stack runs, *its*
final representation (not any patch's) feeds the classification head —
a dedicated "summary" position that's free to attend to (and be attended
by) every patch, rather than forcing one arbitrary patch to double as
the whole image's summary. Prepending a fixed-size learnable vector to a
per-batch sequence is exactly what motivated
[`Concat`](16-concat.md):

```python
cls_tokens = self.cls_token + Tensor(np.zeros((batch, 1, embed_dim)))
x = F.concat([cls_tokens, x], axis=1)
```

(the `+ zeros` broadcasts the single learned CLS vector to every item in
the batch before concatenation, since `Concat` itself doesn't broadcast —
every non-concatenated axis has to already match.)

## Learned positional embedding, not sinusoidal

Unlike the [Shakespeare example](15-shakespeare-example.md)'s fixed
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
gradient back down (the exact mechanism [Attention](10-multi-head-attention.md)'s
batched `MatMul` relies on too). Patches have no inherent left-to-right
order the way text tokens do, and unlike text there's no need to
generalize to unseen sequence lengths at inference time (image size,
hence patch count, is fixed per model) — so a learned table is the
simpler and empirically stronger choice for ViT specifically.

## No mask

`TransformerBlock.forward(x, mask=None)` runs with no mask at all here —
unlike causal language modeling, an image classifier has no
"future" to hide from a patch. Every patch (and the CLS token) may
freely attend to every other patch.

## Picking out the CLS token without a new indexing op

After the Transformer stack and a final `LayerNorm`, only the CLS
token's output (position 0) goes to the classification head — but
nabla has no `Tensor.__getitem__`/slicing op yet. Rather than add one
just for this, `VisionTransformer` reuses `MatMul`, which already has a
backward: a fixed (non-learnable, `requires_grad=False`) one-hot row
picks out position 0 by construction —

```python
selector = np.zeros((1, 1, num_patches + 1))
selector[0, 0, 0] = 1.0
self._cls_selector = Tensor(selector)
...
cls_out = F.matmul(self._cls_selector, x).reshape((batch, -1))
```

`selector @ x` sums over the sequence axis with weight 1 at position 0
and 0 everywhere else, so the sum is exactly `x[:, 0, :]`. Same trick as
everywhere else in this codebase: prefer composing an already-differentiable
op over hand-deriving a new one when the composition is this direct.

## Testing

[`tests/test_nn_patch_embedding.py`](../tests/test_nn_patch_embedding.py)
checks patch-count/shape math, rejects non-divisible patch sizes, and
verifies patches are spatially independent (zeroing one patch's pixels
must not change any other patch's embedding). [`tests/test_nn_vit.py`](../tests/test_nn_vit.py)
checks end-to-end forward shape (including a multi-channel/RGB case),
that gradients reach every parameter including `cls_token` and
`pos_embedding`, that the fixed CLS selector is correctly excluded from
`parameters()`, and that `train()`/`eval()` propagates down into the
block stack's dropout layers.
