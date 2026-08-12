# Dropout

## The idea

During training, a network can overfit by relying too heavily on specific
combinations of neurons — effectively memorizing the training set instead of
learning features that generalize. Dropout counters this by randomly zeroing
a fraction `p` of activations on every forward pass, forcing the network to
not depend on any single neuron being present. At test time no randomness is
involved — the full network is used.

Unlike every other op documented so far, Dropout has no learnable
parameters and, more unusually, is **not a pure function of its input**: the
same `x` produces a different output on every call, because
[`ops/dropout.py`](../src/nabla/ops/dropout.py) draws a fresh random mask
each time. That randomness is also exactly why nabla splits training and
eval behavior between two layers, the same pattern as
[BatchNorm2D](04-batchnorm.md):

- **`ops/dropout.py`** ([`Dropout`](../src/nabla/ops/dropout.py)) is the
  differentiable masking operation — always stochastic, meant for training
  only. This doc covers that.
- **`nn/dropout.py`** ([`Dropout`](../src/nabla/nn/dropout.py)) is the
  `Module` wrapper: in training mode it calls the op above, in eval mode it
  just returns `x` unchanged — no `Function` involved at all.

## The math

For input $x$ of any shape, each element independently survives with
probability $(1-p)$:

$$
m_i \sim \text{Bernoulli}(1-p), \qquad \text{out}_i = \frac{m_i}{1-p} \cdot x_i
$$

The $\frac{1}{1-p}$ factor is what makes this **inverted** dropout. Without
it, $\mathbb{E}[\text{out}_i] = (1-p) \cdot x_i$ — the expected activation
would shrink by a factor of $(1-p)$, so eval mode would need to rescale by
that same factor to match. Scaling up at training time instead keeps the
expected value unchanged in *either* mode:

$$
\mathbb{E}[\text{out}_i] = (1-p) \cdot \frac{1}{1-p} \cdot x_i = x_i
$$

which is exactly why eval mode can be a plain, cost-free passthrough — all
the correction already happened during training.

```python
keep = np.random.rand(*x.data.shape) < (1 - self.p)
self.mask = keep.astype(x.data.dtype) / (1 - self.p)
return x.data * self.mask
```

`p = 0` is not a special case: every element survives, `mask` is uniformly
`1 / 1 = 1`, and `out == x` falls out of the formula on its own.

## Backward

Once `mask` is fixed for a given forward call, $\text{out} = x \cdot
\text{mask}$ is a plain elementwise multiplication — the same shape of
gradient as [ReLU](01-autodiff.md), which also masks based on something
computed in `forward`:

$$
\frac{\partial \, \text{out}_i}{\partial \, x_i} = \text{mask}_i
\qquad\Rightarrow\qquad
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \text{out}_i} \cdot \text{mask}_i
$$

```python
def backward(self, grad_output: NDArray) -> tuple[NDArray]:
    return (grad_output * self.mask,)
```

The gradient flows unchanged through surviving positions (scaled by
$1/(1-p)$, same as the forward pass) and is zeroed exactly where the
activation itself was zeroed — a dropped neuron contributes nothing in
either direction.

## Testing something stochastic

Every other op in nabla is checked against a numerical gradient by calling
`forward()` repeatedly with small perturbations of one input element at a
time. That doesn't directly work here: `forward()` draws a *new* random mask
on every call, so two calls a few `eps` apart wouldn't be comparing the same
function.

[`tests/test_dropout_ops.py`](../tests/test_dropout_ops.py) works around
this by sampling the mask once (a real `Dropout.apply(x, p=...)` call) and
reusing that fixed mask for every perturbed evaluation:

```python
out = Dropout.apply(x, p=0.3)
mask = out._ctx.mask

def forward_fixed_mask():
    return x.data * mask
```

This checks the same thing a numerical gradient check always checks —
"does `backward` compute the derivative of what `forward` actually
computed" — just holding the random part constant so the comparison is
well-defined. The remaining tests check the statistical properties directly
instead: roughly `p` fraction of elements are zero, survivors are scaled by
exactly $1/(1-p)$, and `p=0` leaves `x` untouched.

## Eval mode is a no-op, not a second Function

[`nn/dropout.py`](../src/nabla/nn/dropout.py) doesn't call into
`ops/dropout.py` at all in eval mode:

```python
def forward(self, x: Tensor) -> Tensor:
    if self.training:
        return DropoutFunction.apply(x, p=self.p)
    return x
```

There's no running-statistics bookkeeping to do (unlike BatchNorm) — inverted
dropout means the training-time scaling already accounts for eval-time
behavior, so passing `x` straight through is the entire eval-mode
implementation. Both modes are checked in
[`tests/test_nn_dropout.py`](../tests/test_nn_dropout.py), including that
eval mode returns the exact same `Tensor` object rather than a new one.
