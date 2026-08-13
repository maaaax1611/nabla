# LayerNorm

## The idea

Like [BatchNorm2D](04-batchnorm.md), LayerNorm re-centers and re-scales
activations to zero mean and unit variance, then lets the network learn its
own scale and shift (`gamma`, `beta`) back on top. The difference is *which*
elements get averaged together to compute that mean/variance:

- BatchNorm2D normalizes **per channel**, over every other sample in the
  batch (+ spatial axes) — so it needs a reasonably large batch to get a
  stable estimate, and behaves differently at train vs. eval time.
- LayerNorm normalizes **per sample**, over that sample's own feature axis
  — every sample is normalized independently of every other one.

That independence is exactly why LayerNorm is the normalization of choice
for Transformers: sequence batches often mix samples of different
effective lengths (padding), and you don't want one sample's statistics
leaking into another's, or the normalization to depend on batch size at
all. It also means LayerNorm needs no running statistics and behaves
identically whether the model is training or not — see
["No train/eval split"](#no-traineval-split) below.

## The math

For input $x$ of shape `(*, \text{features})` — any number of leading axes,
with `features` last — mean and variance are computed **per sample**, over
just that last axis, using $N = \text{features}$ values:

$$
\mu = \frac{1}{N}\sum_k x_k \qquad \sigma^2 = \frac{1}{N}\sum_k (x_k - \mu)^2
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \qquad \text{out} = \gamma \cdot \hat{x} + \beta
$$

This is the *identical* formula to BatchNorm2D — only which axis gets
reduced over changes. `gamma`/`beta` still have shape `(features,)`, but
because `features` is already the *last* axis of `x`, they broadcast
against it directly with no `.reshape(...)` needed (BatchNorm2D's channels
sit on the second axis of `(batch, C, H, W)`, which is why *that* op needs
to reshape `gamma`/`beta` to `(1, C, 1, 1)` first).

## Backward: same graph, different axis

Because the math is identical, so is the computational graph — decomposing
`out = gamma * x_hat + beta` into atomic operations produces the exact same
nine-node graph documented in
[BatchNorm2D's backward derivation](04-batchnorm.md#backward-derived-from-the-computational-graph):
`x → mu`, `x - mu → xmu`, `xmu² → sq`, `mean(sq) → var`, and so on through
`sqrtvar`, `ivar`, `xhat`, `gammax`, `out` — with `x` still forking into two
paths (direct, through `xmu`, and indirect, through `mu`) that get summed
at the very end. The only thing that changes walking that graph backward is
*which axis* every `sum`/`mean` reduces over — `axis=(0, 2, 3)` becomes
`axis=-1` — and what `N` means: batch·H·W becomes just `features`.

```python
leading_axes = tuple(range(grad_output.ndim - 1))
grad_beta = np.sum(grad_output, axis=leading_axes)
grad_gamma = np.sum(grad_output * self.x_hat, axis=leading_axes)

grad_x_hat = grad_output * gamma.data

grad_var = np.sum(
    grad_x_hat * self.x_centered * -0.5 * (self.var + self.eps) ** -1.5,
    axis=-1,
    keepdims=True,
)
grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)
grad_mean += grad_var * np.sum(-2 * self.x_centered, axis=-1, keepdims=True) / self.N

grad_x = grad_x_hat / np.sqrt(self.var + self.eps)
grad_x += grad_var * 2 * self.x_centered / self.N
grad_x += grad_mean / self.N
```

Note `grad_beta`/`grad_gamma` sum over `leading_axes` (every axis *except*
the last one) rather than a fixed `axis=(0, 2, 3)` — LayerNorm has to work
for any number of leading axes (a plain `(batch, features)` input, or a
Transformer's `(batch, seq_len, features)`), so which axes count as
"leading" depends on the input's rank, computed once from
`grad_output.ndim` rather than hardcoded.

## Implementation

[`ops/layernorm.py`](../src/nabla/ops/layernorm.py) caches exactly what the
backward table above needs — `x_centered`, `x_hat`, `var`, and `N` — the
same "what does forward need to *cache* for backward" question that came up
for [BatchNorm2D](04-batchnorm.md#bug-we-hit), just answered once and
gotten right this time. Correctness is checked the same way as every other
op in the library: a naive reference implementation for forward, and
numerical gradient checks for backward — including a `(batch, seq_len,
features)` case to specifically cover the Transformer-shaped input LayerNorm
exists for, in
[`tests/test_layernorm_ops.py`](../tests/test_layernorm_ops.py).

## No train/eval split

Unlike [`nn/batchnorm.py`](../src/nabla/nn/batchnorm.py),
[`nn/layernorm.py`](../src/nabla/nn/layernorm.py) has no `running_mean` /
`running_var`, and its `forward` doesn't branch on `self.training` at all
— it always calls the same `Function`:

```python
def forward(self, x: Tensor) -> Tensor:
    return LayerNormFunction.apply(x, self.gamma, self.beta, eps=self.eps)
```

That's a direct consequence of normalizing per sample instead of per batch:
there's no batch-level statistic to smooth into a running average in the
first place, so eval mode has nothing different to do than training mode.
