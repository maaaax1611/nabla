# BCEWithLogitsLoss

## Overview

`BCEWithLogitsLoss` computes binary cross-entropy directly from raw
logits, without a separate sigmoid step. Used with [`UNet`](28-unet.md)
(which returns logits, not probabilities) and typically combined with
[`DiceLoss`](27-dice-loss.md) for segmentation - see the "Why combine
with Dice" section below.

## Math

Binary cross-entropy, given probability `p = sigmoid(x)` and label `y`:

$$
\text{loss} = -\big[y \log(p) + (1-y)\log(1-p)\big]
$$

Computing `sigmoid(x)` and then `log(...)` separately is numerically
unsafe: a confidently wrong prediction pushes `x` far enough that
`sigmoid(x)` rounds to exactly `0.0` or `1.0` in float32, and `log(0)` is
`-inf`. Substituting `p = sigmoid(x)` into the formula above and
simplifying algebraically (using `log(sigmoid(x)) = -log(1+e^{-x})`)
gives a form that only ever evaluates `log` on `1 + e^{-|x|}`, a value
always in `(1, 2]`:

$$
\text{loss} = \max(x, 0) - xy + \log\!\left(1 + e^{-|x|}\right)
$$

This is mathematically identical to the first formula for every finite
`x`, but never overflows or underflows regardless of how large `|x|`
gets.

## Backward

Differentiating the stable form with respect to `x` collapses back down
to the same clean shape as [`SoftmaxCrossEntropy`](05-softmax-cross-entropy.md)'s
gradient - a fused sigmoid+cross-entropy Function trading its
complexity for the simplest possible backward pass:

$$
\frac{\partial \, \text{loss}}{\partial x} = \text{sigmoid}(x) - y
$$

divided by the number of elements the forward mean was taken over.

## Implementation

```python
def forward(self, logits: Tensor, targets: Tensor) -> NDArray:
    self.save_for_backward(logits, targets)
    xp = get_array_module(logits.data)
    x, y = logits.data, targets.data

    loss = xp.maximum(x, 0) - x * y + xp.log1p(xp.exp(-xp.abs(x)))
    self.num_elements = x.size
    return loss.mean()

def backward(self, grad_output: NDArray) -> tuple[NDArray, NDArray]:
    logits, targets = self.saved_tensors
    xp = get_array_module(grad_output)

    sigmoid_x = 1 / (1 + xp.exp(-logits.data))
    grad_logits = (sigmoid_x - targets.data) * (grad_output / self.num_elements)

    grad_targets = xp.zeros_like(targets.data)
    return grad_logits, grad_targets
```

`xp.log1p` (log(1+z)) rather than `xp.log(1 + z)` - more accurate than
composing `+` and `log` separately when `z` is small, which
`e^{-|x|}` frequently is for confidently-correct predictions.

`targets` is a fixed 0/1 label, not differentiable - same reasoning as
[`SoftmaxCrossEntropy`](05-softmax-cross-entropy.md#targets-isnt-differentiable)
and [`DiceLoss`](27-dice-loss.md).

## Why combine with Dice

Pure Dice loss has a well-documented failure mode on extremely
imbalanced segmentation targets (BraTS tumor masks are ~0.25% positive
pixels): if the model predicts a probability near 0 everywhere, the
intersection term in Dice's numerator stays near 0, and its gradient can
be too small to reliably escape that "predict all background" state.
BCE's gradient doesn't have this problem - `sigmoid(x) - y` pushes every
pixel toward its own label regardless of how rare the positive class is
- so a common combination (used, among others, in the PyTorch reference
this project's BraTS pipeline was validated against) is a weighted sum:

```python
loss = bce_criterion(logits, targets) + 5.0 * dice_criterion(logits.sigmoid(), targets)
```

BCE breaks the initial plateau; Dice still drives the actual
segmentation-overlap quality once training is past it.

## Testing

[`tests/test_loss.py::TestBCEWithLogitsLoss`](../tests/test_loss.py)
checks forward against a naive (unstable) reference on well-behaved
inputs, low/high loss for confident correct/wrong predictions, that
extreme logits (`±100`) stay finite (where the naive reference would
hit `log(0)`), backward against a numerical gradient, the closed-form
`sigmoid(x) - y` gradient directly, that `targets`'s gradient is zero,
and the shape-mismatch validation error.
