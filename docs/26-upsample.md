# Upsample

## Why this op, and why now

The first work on `feature/segmentation` is a U-Net for BraTS brain-tumor
segmentation. A U-Net's encoder shrinks spatial resolution with
`MaxPool2D` (already in nabla); its decoder needs to grow it back so the
final output matches the input's `H x W`. That's `Upsample`.

Two standard ways to grow resolution back: nearest-neighbor upsampling
(no parameters, simple gradient) or a transposed convolution (learnable,
but a more involved backward derivation). The original U-Net paper itself
uses upsampling + a regular `Conv2D` afterward rather than a transposed
conv, and that also fits nabla's "one op, one clear derivation" pattern
better — so nearest-neighbor it is. A learnable transposed-conv variant is
a plausible future addition, not a gap in this one.

## The math

$$
\text{out}[b, c, i, j] = x[b, c, \lfloor i / s \rfloor, \lfloor j / s \rfloor]
$$

Forward duplicates every pixel into an `s x s` block, where `s` is the
integer scale factor:

```python
def forward(self, x: Tensor) -> NDArray:
    xp = get_array_module(x.data)
    self.input_shape = x.data.shape
    out = xp.repeat(x.data, self.scale, axis=2)
    out = xp.repeat(out, self.scale, axis=3)
    return out
```

`xp.repeat(..., axis=2)` repeats along `H`, then the same along `W` —
together that turns each input pixel into an `s x s` block of identical
values in the output, exactly matching the formula above.

## Backward — the reverse of "duplicate" is "sum"

Every input pixel contributed to `s * s` output pixels in the forward
pass, so its gradient is the *sum* of all `s * s` of their gradients —
the same "one value feeds many outputs -> sum their gradients" pattern
`ops/broadcast.py` uses for broadcasting, just over spatial blocks instead
of broadcast axes:

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray]:
    xp = get_array_module(grad_output)
    b, c, h, w = self.input_shape
    grad_reshaped = grad_output.reshape(b, c, h, self.scale, w, self.scale)
    grad_input = grad_reshaped.sum(axis=(3, 5))
    return (grad_input,)
```

The `reshape` is the key trick, and it's the exact mirror of forward's
`repeat`: a `(B, C, H*s, W*s)` array is reinterpreted as
`(B, C, H, s, W, s)` — splitting each spatial axis back into "which block"
and "which position inside the block" — without moving any data, since
reshape only reinterprets an already-contiguous layout. Summing over axes
`3` and `5` (the "inside the block" axes) then collapses each `s x s`
block of gradients back down to the single value the one input pixel
that produced it needs.


## No learnable parameters

Like `Dropout`, `nn/upsample.py` wraps the op with no weights of its own
— `Module.parameters()` returns nothing for it, since there's nothing to
train:

```python
class Upsample(Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x.upsample(scale=self.scale)
```

## Testing

[`tests/test_upsample_ops.py`](../tests/test_upsample_ops.py) checks
forward against a hand-computed block duplication, backward against both
a hand-computed block-sum and a numerical gradient, the `scale=1` no-op
edge case, and that the `nn.Upsample` wrapper forwards correctly with an
empty parameter list.
