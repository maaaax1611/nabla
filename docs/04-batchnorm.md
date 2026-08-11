# BatchNorm2D

## The idea

As a network trains, the distribution of activations flowing into each
layer keeps shifting (its inputs change every time the layers before it
update). Batch normalization re-centers and re-scales each channel's
activations to zero mean and unit variance *within the current batch*, then
lets the network learn its own preferred scale and shift back (`gamma`,
`beta`) on top of that. In practice this stabilizes and speeds up training
considerably.

nabla splits this into two layers, deliberately:

- **`ops/batchnorm.py`** ([`BatchNorm2D`](../src/nabla/ops/batchnorm.py)) is
  the differentiable normalization *using the current batch's statistics*
  (training mode) — this doc covers that.
- **`nn/batchnorm.py`**
  ([`BatchNorm2D`](../src/nabla/nn/batchnorm.py)) is the `Module` wrapper:
  it holds `gamma`/`beta` as trainable parameters, calls the op above in
  training mode, tracks a running average of the batch statistics
  (`running_mean`, `running_var`), and switches to normalizing with *those*
  running statistics in eval mode — via ordinary `Tensor` arithmetic, not a
  second `Function` (see the note at the end).

## The math

For input $x$ of shape `(batch, C, H, W)`, mean and variance are computed
**per channel**, over every other axis (`batch`, `H`, `W`) — so each of the
$C$ channels gets normalized independently, using $N = \text{batch} \cdot H
\cdot W$ values:

$$
\mu_c = \frac{1}{N}\sum x_c \qquad \sigma^2_c = \frac{1}{N}\sum (x_c - \mu_c)^2
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \qquad \text{out} = \gamma \cdot \hat{x} + \beta
$$

($\varepsilon$ is a small constant purely to avoid dividing by zero.) Note
this uses the **biased** variance (divide by $N$, not $N{-}1$) — the same
convention PyTorch and every other framework uses for the normalization
itself.

## Backward, derived step by step

This is the one op in nabla where the backward pass genuinely needs the
full chain rule, because $\mu$ and $\sigma^2$ are each themselves functions
of *every* element of $x$ — so $\partial L/\partial x$ has to account for
$x$'s effect on the output both directly (through $\hat x$) and indirectly
(through $\mu$ and $\sigma^2$).

**1. The two easy ones.** $\beta$ and $\gamma$ only affect the output
through simple addition/multiplication, so their gradients are the
straightforward sums:

$$
\frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial \text{out}} \qquad
\frac{\partial L}{\partial \gamma} = \sum \frac{\partial L}{\partial \text{out}} \cdot \hat{x}
$$

**2. Gradient w.r.t. $\hat x$.** Just undo the scale by $\gamma$:

$$
\frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial \text{out}} \cdot \gamma
$$

**3. Gradient w.r.t. $\sigma^2$.** $\hat x$ depends on $\sigma^2$ through
$(\sigma^2 + \varepsilon)^{-1/2}$. Every element of $\hat x$ (across the
whole channel) depends on the same $\sigma^2$, so their contributions sum:

$$
\frac{\partial L}{\partial \sigma^2} = \sum \frac{\partial L}{\partial \hat{x}} \cdot (x - \mu) \cdot \left(-\tfrac{1}{2}\right)(\sigma^2 + \varepsilon)^{-3/2}
$$

**4. Gradient w.r.t. $\mu$.** This is the subtle one — $\mu$ affects
$\hat{x}$ *and* $\sigma^2$ (which itself depends on $\mu$), so there are
**two paths** back to $\mu$ and both need to be added:

$$
\frac{\partial L}{\partial \mu} =
\underbrace{\sum \frac{\partial L}{\partial \hat{x}} \cdot \frac{-1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{direct path, through } \hat x}
\;+\;
\underbrace{\frac{\partial L}{\partial \sigma^2} \cdot \frac{1}{N}\sum -2(x-\mu)}_{\text{indirect path, through } \sigma^2}
$$

**5. Gradient w.r.t. $x$.** Finally, $x$ feeds into $\hat x$ directly *and*
into both $\mu$ and $\sigma^2$ (each of which is itself an average over all
$N$ elements, so each element of $x$ gets $1/N$ of their gradient):

$$
\frac{\partial L}{\partial x} =
\underbrace{\frac{\partial L}{\partial \hat{x}} \cdot \frac{1}{\sqrt{\sigma^2+\varepsilon}}}_{\text{direct}}
\;+\;
\underbrace{\frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x-\mu)}{N}}_{\text{through } \sigma^2}
\;+\;
\underbrace{\frac{\partial L}{\partial \mu} \cdot \frac{1}{N}}_{\text{through } \mu}
$$

## Implementation

Each of those five steps is one block in
[`ops/batchnorm.py`](../src/nabla/ops/batchnorm.py):

```python
grad_beta  = np.sum(grad_output, axis=(0, 2, 3))
grad_gamma = np.sum(grad_output * self.x_hat, axis=(0, 2, 3))

grad_x_hat = grad_output * gamma_reshaped

grad_var = np.sum(grad_x_hat * self.x_centered * -0.5 * (self.batch_var + self.eps) ** -1.5,
                   axis=(0, 2, 3), keepdims=True)

grad_mean  = np.sum(grad_x_hat * -1 / np.sqrt(self.batch_var + self.eps), axis=(0, 2, 3), keepdims=True)
grad_mean += grad_var * np.mean(-2 * self.x_centered, axis=(0, 2, 3), keepdims=True)

grad_x  = grad_x_hat / np.sqrt(self.batch_var + self.eps)
grad_x += grad_var * 2 * self.x_centered / self.N
grad_x += grad_mean / self.N
```

`self.x_centered` (i.e. $x - \mu$) and `self.batch_var` are cached in
`forward` specifically because `backward` needs them again here — that's
the general pattern every `Function` in nabla follows (see
[Autodiff basics](01-autodiff.md)).

> **Bug we hit:** `forward` originally never set `self.N`
> (`batch * H * W`, the count backward divides by in steps 4 and 5 above) —
> an easy thing to forget since it's not needed anywhere in forward itself,
> only in backward. First call to `backward()` raised
> `AttributeError: 'BatchNorm2D' object has no attribute 'N'`. A reminder
> that "what does forward need to *cache* for backward" is a separate
> question from "what does forward need to *compute* its own output".

## Eval mode reuses ordinary ops, not a second Function

`nn/batchnorm.py`'s eval-mode path normalizes with the running statistics
using plain `Tensor` arithmetic instead of calling into the `Function`
above:

```python
mean = Tensor(self.running_mean.reshape(1, -1, 1, 1))
std  = Tensor(np.sqrt(self.running_var + self.eps).reshape(1, -1, 1, 1))
gamma = self.gamma.reshape((1, self.num_features, 1, 1))
beta  = self.beta.reshape((1, self.num_features, 1, 1))
return (x - mean) / std * gamma + beta
```

This works because `running_mean`/`running_var` are *constants* at this
point (no gradient needs to flow into them — they're an exponential moving
average maintained outside of autodiff entirely), so there's no need for a
custom backward derivation at all: ordinary `Subtract`, `Divide`,
`Multiply`, `Add` compose into exactly the right gradient for `x`, `gamma`,
and `beta` automatically. This is also precisely what first exposed the
`unbroadcast` bug described in [Autodiff basics](01-autodiff.md) —
`(1, C, 1, 1)` broadcasting against `(batch, C, H, W)` at equal `ndim` was a
case nothing had exercised before.

Both modes — including the running-statistics update and the `train()` /
`eval()` switch — are checked in
[`tests/test_batchnorm_ops.py`](../tests/test_batchnorm_ops.py) and
[`tests/test_nn_batchnorm.py`](../tests/test_nn_batchnorm.py).
