# Dice Loss

## Why this loss, not cross-entropy

Segmentation masks are usually mostly background: a brain-tumor mask
might be 1-2% "tumor" pixels and 98%+ "not tumor". Per-pixel
cross-entropy would let a model get a great loss just by predicting
"background everywhere" — the overwhelming class dominates the average.
Dice loss instead measures *overlap* between prediction and ground truth
directly, which stays meaningful even when one class is rare.

## The math

$$
\text{Dice} = \frac{2 \sum(p \cdot y) + \epsilon}{\sum p + \sum y + \epsilon}
\qquad
\text{loss} = 1 - \text{Dice}
$$

summed over every axis except batch (so each sample gets its own Dice
score, then the loss is the mean over the batch). `p` is the predicted
probability mask (already sigmoid'd — this Function applies no
activation), `y` the binary ground-truth mask.

```python
def forward(self, probs: Tensor, target: Tensor) -> NDArray:
    xp = get_array_module(probs.data)
    self.save_for_backward(probs, target)
    axes = tuple(range(1, probs.data.ndim))
    self.intersection = (probs.data * target.data).sum(axis=axes)
    self.union = xp.sum(probs.data, axis=axes) + xp.sum(target.data, axis=axes)
    dice = (2 * self.intersection + self.eps) / (self.union + self.eps)
    return (1 - dice).mean()
```

`self.intersection`/`self.union` are cached (not just local variables)
because backward needs them again, the same pattern `SoftmaxCrossEntropy`
uses for `self.probs`.

### Where `eps` goes matters

`eps` isn't just "a small number to avoid dividing by zero" — its exact
placement changes what the loss does at the edge case both masks are
completely empty (no tumor present, correctly predicted as such — which
should score as a *perfect* match, loss = 0). A first attempt added `eps`
to `intersection` before doubling it (`intersection = sum(p*y) + eps`,
then `2 * intersection`), which doubles `eps` in the numerator relative
to the denominator; for two empty masks that gives `dice = 2*eps/eps = 2`
and `loss = 1 - 2 = -1` — a nonsensical negative loss for a perfect
match. `eps` has to be added exactly once, after the `2 *`, in both
numerator and denominator equally, as in the formula above — then two
empty masks give `dice = eps/eps = 1`, `loss = 0`, correctly.

## Backward — quotient rule, then broadcasting it back to pixel shape

Dice is a ratio `u/v` per sample, with `u = 2·intersection + eps` and
`v = union + eps`. Differentiating each per-pixel `p_i` with the quotient
rule (`d(u/v)/dp_i = (u'_i·v - u·v'_i) / v²`), where `u'_i = 2·y_i` and
`v'_i = 1`:

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
    p_tensor, y_tensor = self.saved_tensors
    p, y = p_tensor.data, y_tensor.data
    I, U, eps = self.intersection, self.union, self.eps
    xp = get_array_module(p)

    batch = p.shape[0]
    shape = (batch,) + (1,) * (p.ndim - 1)
    I = I.reshape(shape)
    U = U.reshape(shape)

    numerator = (2 * I + eps) - 2 * y * (U + eps)
    grad_p = numerator / (U + eps) ** 2 / batch

    grad_target = xp.zeros_like(y)
    return grad_output * grad_p, grad_target
```

Two things worth pointing out:

- **The reshape.** `I` and `U` are `(batch,)` — one scalar per sample —
  but `y`/`p` are `(batch, H, W, ...)`. Multiplying `y * (U + eps)`
  directly would try to broadcast `(batch,)` against `(batch, H, W)`,
  which numpy aligns from the *trailing* axes — `U`'s only axis would
  try to line up with `W`, not batch, either raising a shape error or
  (worse, if the sizes happened to coincide) silently multiplying by the
  wrong sample's union. Reshaping to `(batch, 1, 1, ...)` first forces
  numpy to broadcast each sample's scalar `I`/`U` across that sample's
  own pixels, and works for any number of spatial axes since
  `(1,) * (p.ndim - 1)` grows with `p`'s rank.
- **The `/ batch`.** Forward returns `(1 - dice).mean()` — the mean over
  the batch. Differentiating a mean introduces a `1/batch_size` factor
  (`d(mean(x))/dx_i = 1/n`) that's easy to forget since it doesn't show
  up anywhere in the per-sample Dice formula itself.

`target` isn't differentiable (same reasoning as `targets` in
[`SoftmaxCrossEntropy`](05-softmax-cross-entropy.md#targets-isnt-differentiable)),
so its gradient is all zeros.

Verified against a numerical gradient (central difference) on a
non-trivial `(2, 3, 3)` batch — max absolute difference ~5e-11.

## Testing

[`tests/test_loss.py::TestDiceLoss`](../tests/test_loss.py) checks the
three sanity anchors (perfect overlap → loss ≈ 0, two empty masks →
loss ≈ 0, complete mismatch → loss ≈ 1), forward against a naive NumPy
reference, backward against a numerical gradient, that `target`'s
gradient is all zeros, and the shape-mismatch validation error.
