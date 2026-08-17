# Concat

## Why we need it

Every op so far takes a fixed, small number of tensors. The
[Vision Transformer](17-vision-transformer.md) needs something new: a
learnable **CLS token** prepended to the sequence of patch embeddings
before the Transformer stack runs over it — the same trick BERT and ViT
both use, a dummy "summary" position whose final output feeds the
classification head. That's concatenation along the sequence axis, and
nothing in nabla could do that yet.

## The op

[`ops/concat.py`](../src/nabla/ops/concat.py)'s `Concat` is the first op
in nabla that takes a *variable* number of tensors rather than one or
two — `Function.apply(*inputs, **kwargs)` and
`Function.backward(...) -> tuple[...]` already supported this, they just
hadn't been exercised by an op that needed it yet.

```python
def forward(self, *tensors):
    self.save_for_backward(*tensors)
    return np.concatenate([t.data for t in tensors], axis=self.axis)
```

## The backward pass

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

## The off-by-one bug that came up

An earlier version computed `split_indices` as
`cumsum([0] + tensor_sizes)[:-1]`. Prepending that `0` shifts every cut
point down by one slot and, worse, adds a phantom leading split: for
three tensors, `np.split` returns *four* pieces (`[0:0]`, `[0:3]`,
`[3:8]`, `[8:10]`) — an empty array first, then everything after it
misaligned with the wrong input. Numerical gradient checking on all
three inputs caught it immediately. The fix was just to drop the
leading `0` — `cumsum(tensor_sizes)[:-1]` already starts at the first
real cut point.

## Testing

[`tests/test_concat_ops.py`](../tests/test_concat_ops.py) checks forward
shapes and values for two and three tensors, that `backward` routes each
slice of `grad_output` back to exactly the right input (checked both by
direct slicing and by central-difference numerical gradient checking),
and that gradient shapes always match their input's shape even when
inputs have very different sizes along the concatenation axis.
