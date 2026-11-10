---
title: "Training: optimizers, regularization, schedules"
description: "SGD and Adam, weight decay, dropout, learning-rate schedules and gradient clipping."
---

# Training: optimizers, regularization, schedules

`loss.backward()` leaves a gradient on every parameter. This post is about
what happens next: how an optimizer turns gradients into updates, and the
pieces around it that decide whether training is stable — regularization,
dropout, learning-rate schedules and gradient clipping.

## Mini-batches

[`DataLoader`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/data/dataloader.py) shuffles the sample indices
once per epoch and yields `(X, y)` slices of `batch_size`. Averaging the
loss over a batch gives a noisy but unbiased estimate of the full gradient
at a fraction of the cost; the noise shrinks with $1/\sqrt{\text{batch size}}$,
which is why doubling the batch size does not halve the number of steps you
need.

## SGD

The whole algorithm ([`optim/sgd.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/optim/sgd.py)):

$$
\theta \leftarrow \theta - \eta\,g, \qquad g = \nabla_\theta L
$$

```python
for param in self.parameters:
    if param.grad is not None:
        param.data -= self.lr * self._regularized_grad(param)
```

The update writes to `param.data` directly instead of going through
`Tensor` arithmetic. That is deliberate: the update is not part of the
function being differentiated, so it must not show up in the graph.

## Adam

Plain SGD uses one step size for every parameter. Adam
([`optim/adam.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/optim/adam.py)) keeps two running averages
per parameter and uses them to scale each coordinate individually:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\,g_t\\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

$m$ is a running mean of the gradient, $v$ a running mean of its square;
the third line is the bias correction.

Two things are worth understanding rather than memorizing:

- **The ratio $\hat{m} / \sqrt{\hat{v}}$ is roughly scale-free.** Multiply
  every gradient of one parameter by 100 and both numerator and denominator
  grow by 100. Each coordinate therefore moves about $\eta$ per step,
  regardless of how large its raw gradient is — which is why the same
  `lr=1e-3` works across very different layers.
- **Bias correction fixes the cold start.** $m$ and $v$ start at zero. After
  the first step $m_1 = (1 - \beta_1)\,g_1 = 0.1\,g_1$, ten times too small
  as an estimate of the mean. Dividing by $1 - \beta_1^t$ undoes exactly
  that, and the correction fades to 1 as $t$ grows. With $\beta_2 = 0.999$
  the second moment needs about a thousand steps to warm up, so without the
  correction the first updates would be far too large.

## Weight decay as a gradient term

An L2 penalty $\frac{\lambda}{2}\lVert\theta\rVert^2$ added to the loss
contributes $\lambda\,\theta$ to the gradient; an L1 penalty
$\lambda\lVert\theta\rVert_1$ contributes $\lambda\,\text{sign}(\theta)$.
nabla never adds the penalty to the loss tensor at all. A
[`Regularizer`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/regularizers.py) returns that gradient
contribution directly, and the optimizer adds it just before the update:

```python
def _regularized_grad(self, param):
    grad = param.grad
    for reg in self.regularizers:
        grad = grad + reg.apply(param)
    return grad
```

`param.grad` itself is left untouched, so logging and gradient clipping
still see the raw gradient.

> **Caveat with Adam:** because the decay term is added to the gradient
> *before* Adam's rescaling, parameters with large gradients get
> proportionally less decay. That is the classic "L2 regularization" form.
> AdamW decouples the two by applying $\theta \leftarrow \theta - \eta\lambda\theta$
> separately from the adaptive step; nabla does not implement that variant.

## Dropout


During training, a network can overfit by relying too heavily on specific
combinations of neurons — effectively memorizing the training set instead of
learning features that generalize. Dropout counters this by randomly zeroing
a fraction `p` of activations on every forward pass, forcing the network to
not depend on any single neuron being present. At test time no randomness is
involved — the full network is used.

Unlike every other op documented so far, Dropout has no learnable
parameters and, more unusually, is **not a pure function of its input**: the
same `x` produces a different output on every call, because
[`ops/dropout.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/dropout.py) draws a fresh random mask
each time. That randomness is also exactly why nabla splits training and
eval behavior between two layers, the same pattern as
[BatchNorm2D](04-cnn.md#batchnorm2d):

- **`ops/dropout.py`** ([`Dropout`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/dropout.py)) is the
  differentiable masking operation — always stochastic, meant for training
  only. This doc covers that.
- **`nn/dropout.py`** ([`Dropout`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/dropout.py)) is the
  `Module` wrapper: in training mode it calls the op above, in eval mode it
  just returns `x` unchanged — no `Function` involved at all.

### Dropout: the math

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

### Dropout: backward

Once `mask` is fixed for a given forward call, $\text{out} = x \cdot
\text{mask}$ is a plain elementwise multiplication — the same shape of
gradient as [ReLU](01-autograd.md#autodiff-basics), which also masks based on something
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

### Testing something stochastic

Every other op in nabla is checked against a numerical gradient by calling
`forward()` repeatedly with small perturbations of one input element at a
time. That doesn't directly work here: `forward()` draws a *new* random mask
on every call, so two calls a few `eps` apart wouldn't be comparing the same
function.

[`tests/test_dropout_ops.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_dropout_ops.py) works around
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

### Eval mode is a no-op, not a second Function

[`nn/dropout.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/dropout.py) doesn't call into
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
[`tests/test_nn_dropout.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_nn_dropout.py), including that
eval mode returns the exact same `Tensor` object rather than a new one.

## Learning-Rate Schedules

### Why a schedule, not a fixed `lr`

Every optimizer so far (`SGD`, `Adam`) used one constant `lr` for the
whole run. That's fine for the small examples built so far, but two
real problems show up once training gets longer or the model gets
bigger:

- **Early instability.** At step 1, every weight is freshly
  initialized and gradients can be large and noisy - taking a full-size
  step immediately can push the model somewhere it never recovers from.
  This is exactly why "Attention Is All You Need" and essentially every
  Transformer recipe since (GPT-2/3, etc.) **warm up** the learning
  rate: ramp it up gradually instead of starting at full strength.
- **Late-training overshoot.** A learning rate that was right for large
  early gradients tends to be too large once the loss has mostly
  flattened out - it keeps bouncing around a good solution instead of
  settling into it. Decaying the learning rate over the course of
  training addresses this.

### The design: a scheduler mutates `optimizer.lr`, nothing else

[`optim/scheduler.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/optim/scheduler.py)'s `LRScheduler`
doesn't wrap or replace an `Optimizer` - nabla's `Optimizer` already
exposes a single mutable `.lr` attribute (there's no per-parameter-group
learning rate the way PyTorch has), so a scheduler is just an external
object that computes one number per step and writes it there:

```python
class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.get_lr(self.step_count)
```

Subclasses only implement `get_lr(step)`. This keeps the scheduler
completely decoupled from what kind of optimizer it's driving - it
never touches `.step()`, momentum buffers, or anything else.

### `WarmupCosineLR`

The schedule used in [`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py):
linear ramp-up for `warmup_steps`, then cosine decay down to `min_lr` by
`total_steps`:

```python
def get_lr(self, step):
    if step < self.warmup_steps:
        return self.base_lr * step / self.warmup_steps
    if step >= self.total_steps:
        return self.min_lr
    progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return self.min_lr + (self.base_lr - self.min_lr) * cosine
```

The cosine term smoothly interpolates from 1 (at the end of warmup) to
0 (at `total_steps`) - multiplying that against `(base_lr - min_lr)`
and adding `min_lr` back maps it onto exactly the `[min_lr, base_lr]`
range the schedule needs, with no discontinuity where warmup ends and
decay begins (both evaluate to `base_lr` at that boundary).

### `StepLR`

A simpler, more traditional schedule: multiply `lr` by `gamma` every
`step_size` steps. Less common for Transformers, but a standard choice
for CNNs (`examples/mnist_cnn.py` doesn't use it yet, but could):

```python
def get_lr(self, step):
    return self.base_lr * (self.gamma ** (step // self.step_size))
```

#### Testing

[`tests/test_scheduler.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_scheduler.py) checks that
warmup linearly reaches `base_lr` at exactly `warmup_steps`, that cosine
decay reaches `min_lr` at exactly `total_steps` and stays there
afterward, that the schedule is monotonically non-increasing after
warmup ends, that `StepLR` decays by the right power of `gamma` at each
boundary, and that constructing a scheduler with `warmup_steps >
total_steps` is rejected outright rather than producing a nonsensical
schedule.

## Gradient Clipping

### The problem it guards against

Deep stacks of layers - many stacked `TransformerBlock`s, for instance -
can occasionally produce a gradient that's enormous for one batch (a bad
batch, an unlucky combination of activations, early in training before
things have settled). Taking an optimizer step with that gradient can
throw the model's weights somewhere it never recovers from - "exploding
gradients," the same failure mode residual connections
(see the [Transformer block](05-transformer.md#transformer-block)) help prevent on the *forward* pass,
but nothing so far protects against on the *backward* pass.

### Clipping the combined norm, not each gradient independently

[`optim/clip.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/optim/clip.py)'s `clip_grad_norm_`
mirrors PyTorch's `nn.utils.clip_grad_norm_`: it computes **one** L2
norm over every parameter's gradient combined - as if every gradient
were flattened and concatenated into one giant vector - and, if that
combined norm exceeds `max_norm`, scales *every* gradient down by the
same factor:

```python
grads = [p.grad for p in parameters if p.grad is not None]
xp = get_array_module(grads[0])
total_norm = xp.sqrt(sum(xp.sum(g * g) for g in grads))

clip_coef = max_norm / (total_norm + eps)
if clip_coef < 1:
    for p in parameters:
        if p.grad is not None:
            p.grad = p.grad * clip_coef
```

Clipping each gradient *independently* instead (e.g. capping each
parameter's own norm) would change the *direction* of the overall
update - some parameters' gradients would get scaled down more than
others' relative to how they actually compare in magnitude. Using one
combined norm scales everything by the same factor, so the update still
points the same direction through parameter space, just with a smaller
step - exactly what's wanted: rein in the *magnitude* of a bad update
without corrupting its *direction*.

### Where it goes in the training loop

Between `backward()` and `optimizer.step()` - clipping only makes sense
once every gradient has been computed, and before they get used to
update anything:

```python
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The returned `total_norm` (the norm *before* clipping) is worth logging
on its own - see [`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py),
which logs it every step via [`History`](07-engineering.md#logging-metric-tracking). A norm that's
frequently at or near `max_norm` is a useful signal that training is
running hot, even without ever looking at the loss curve.

#### Testing

[`tests/test_clip.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_clip.py) checks the returned norm
against a hand-computed value, that gradients are left untouched when
already under `max_norm`, that clipping preserves direction (the
clipped gradient is a positive scalar multiple of the original), that
the norm is combined correctly across *multiple* parameters (not summed
as if each had its own independent budget), that every parameter gets
scaled by the *same* factor, and that parameters with no gradient are
skipped without error.

---

Previous: [From a neuron to an MLP](02-mlp.md) · Next: [Convolutional networks](04-cnn.md)
