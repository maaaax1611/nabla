---
title: Training
desc: From gradients to a trained model — mini-batches, SGD and Adam, weight decay, dropout, learning-rate schedules, gradient clipping, checkpoints, logging and the GPU.
tags: [SGD, Adam, Weight decay, Dropout, LR schedule, Gradient clipping, Checkpoint, GPU]
order: 1
eyebrow: Training · Optimization
footer: Code excerpts are from <a href="https://github.com/maaaax1611/nabla/tree/main/src/nabla/optim"><code>nabla/optim</code></a>, <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/dropout.py"><code>ops/dropout.py</code></a>, <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/checkpoint.py"><code>checkpoint.py</code></a> and <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/logging.py"><code>logging.py</code></a>, shortened to the essentials. The figures use small toy problems, not the examples.
---

`loss.backward()` leaves a gradient on every parameter. This page is about what happens next: how an optimizer turns gradients into updates, how regularization keeps a model from memorizing its training data, and the pieces around the loop that keep long runs stable and resumable.

## The training loop {#loop group="Optimization" toc="Training loop"}

Every example in nabla trains with the same loop:

```python
for epoch in range(epochs):
    model.train()
    for X, y in DataLoader(X_train, y_train, batch_size=32, shuffle=True):
        loss = criterion(model(X), y)
        model.zero_grad()            # gradients accumulate, so clear them first
        loss.backward()
        optimizer.step()
```

The gradient of the loss over the whole training set would be the exact direction of steepest descent, but computing it for every update is expensive. A **mini-batch** of $B$ samples gives an unbiased estimate of it at a fraction of the cost. The estimate is noisy: its standard deviation falls only with $1/\sqrt{B}$, so four times the batch size halves the noise. In practice many cheap, noisy steps beat a few exact ones. The `DataLoader` shuffles the sample order once per epoch and yields the batches.

## SGD {#sgd}

Stochastic gradient descent moves every parameter a small step against its gradient:

$$
\theta \leftarrow \theta - \eta\, g, \qquad g = \nabla_\theta L
$$

```python
class SGD(Optimizer):
    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * self._regularized_grad(param)
```

The update writes to `param.data` directly instead of using tensor arithmetic. It is not part of the function being differentiated, so it must not be recorded in the graph.

The learning rate $\eta$ is limited by the steepest direction of the loss. Near a minimum the loss is roughly quadratic, $L \approx \tfrac12 \sum_i \lambda_i u_i^2$, and one SGD step multiplies each coordinate $u_i$ by $1 - \eta\lambda_i$. That factor stays below 1 in magnitude only if $\eta < 2/\lambda_\text{max}$. A rate small enough for the steepest direction is then very small for the flattest one, which shows up as zig-zagging across a narrow valley and slow progress along it.

## Adam {#adam}

Adam keeps two running averages per parameter and uses them to give every coordinate its own step size:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

$m$ averages the gradient (momentum), $v$ its square, elementwise. With the defaults $\beta_1 = 0.9$ and $\beta_2 = 0.999$:

- **Each coordinate moves about $\eta$ per step.** Multiply all gradients of one parameter by 100 and both $\hat m$ and $\sqrt{\hat v}$ grow by 100. The step size no longer depends on how large a parameter's gradient happens to be, which is why one learning rate, typically $10^{-3}$ to $3 \cdot 10^{-4}$, works across very different layers.
- **Bias correction fixes the start.** $m$ and $v$ begin at zero, so after the first step $m_1 = 0.1\, g_1$, ten times too small. Dividing by $1 - \beta_1^t$ undoes exactly this and fades to 1 as $t$ grows. $v$ needs about a thousand steps to warm up; without the correction the first updates would be far too large.

::: widget tr-01-optim

> [!abstract] Key points
> - Mini-batches: cheap, noisy, unbiased gradient estimates.
> - SGD: $\theta \leftarrow \theta - \eta g$, stable only for $\eta < 2/\lambda_\text{max}$.
> - Adam: momentum plus a per-coordinate scale; bias correction for the first steps.

## Weight decay {#weight-decay group="Regularization"}

A model with enough parameters can fit its training data perfectly and still do badly on new data. **Regularization** trades a little training loss for better generalization. The simplest form penalizes large weights:

$$
\text{L2:}\;\; L + \frac{\lambda}{2}\lVert\theta\rVert^2 \;\Rightarrow\; g + \lambda\,\theta, \qquad
\text{L1:}\;\; L + \lambda\lVert\theta\rVert_1 \;\Rightarrow\; g + \lambda\,\operatorname{sign}(\theta)
$$

L2 pulls every weight towards zero in proportion to its size; L1 pulls with constant strength and drives small weights to zero. nabla never adds the penalty to the loss tensor. A `Regularizer` returns the extra gradient term directly, and the optimizer adds it just before the update:

```python
def _regularized_grad(self, param):
    grad = param.grad
    for reg in self.regularizers:
        grad = grad + reg.apply(param)        # e.g. weight_decay * param.data
    return grad
```

`param.grad` itself is left untouched, so logging and gradient clipping still see the raw gradient. With Adam this is the classic "L2 regularization": the decay term passes through Adam's per-coordinate scaling, so parameters with large gradients are decayed less. AdamW instead subtracts $\eta\lambda\theta$ separately from the adaptive step; nabla does not implement that variant.

## Dropout {#dropout}

**Dropout** sets every activation to zero with probability $p$ during training, with a new random choice at each step, and scales the survivors by $1/(1-p)$:

$$
y = \frac{m}{1 - p} \odot x, \qquad m_i \sim \operatorname{Bernoulli}(1 - p)
$$

The scaling keeps the expected value unchanged, $\mathbb E[y_i] = (1 - p) \cdot x_i / (1 - p) = x_i$, so at evaluation time the layer can simply pass its input through (*inverted dropout*). Because no unit can count on a particular other unit being present, the network cannot build fragile co-adaptations between units, and it behaves somewhat like an average of many smaller networks.

::: widget tr-02-dropout

The backward pass multiplies by the same mask that the forward pass drew: dropped units get no gradient.

```python
def forward(self, x):
    keep = xp.random.rand(*x.data.shape) < (1 - self.p)
    self.mask = keep.astype(x.data.dtype) / (1 - self.p)
    return x.data * self.mask

def backward(self, grad_output):
    return (grad_output * self.mask,)
```

A random operation cannot be checked against a numerical gradient directly, because every forward call draws a new mask. The test draws the mask once and reuses it for all perturbed evaluations; separate tests check the statistics: about a fraction $p$ of zeros, survivors scaled by exactly $1/(1-p)$.

> [!abstract] Key points
> - Weight decay: an extra gradient term $\lambda\theta$ (L2) or $\lambda\operatorname{sign}\theta$ (L1).
> - Dropout: random mask scaled by $1/(1-p)$ in training, identity in evaluation.
> - Call `model.train()` and `model.eval()`: dropout and batch normalization depend on it.

## Learning-rate schedules {#schedules group="Stability" toc="LR schedules"}

A fixed learning rate is a compromise. Early in training large steps make fast progress; late in training they keep the parameters bouncing around the minimum instead of settling into it. A **schedule** changes the rate over the course of training.

- **Warmup.** At the start, Adam's statistics are not yet reliable and the gradients of a freshly initialized network can be large and erratic. Raising the rate linearly from zero over the first few hundred steps avoids damaging updates; Transformers in particular train much less reliably without it.
- **Cosine decay.** After the warmup the rate follows half a cosine down to a minimum value, falling slowly at first and towards the end.

$$
\eta_t =
\begin{cases}
\eta_\text{base}\, \dfrac{t}{t_\text{warmup}}, & t < t_\text{warmup} \\[2ex]
\eta_\text{min} + (\eta_\text{base} - \eta_\text{min})\, \dfrac{1 + \cos(\pi\, s)}{2}, \quad s = \dfrac{t - t_\text{warmup}}{t_\text{total} - t_\text{warmup}}, & \text{otherwise}
\end{cases}
$$

::: widget tr-03-schedule

A scheduler in nabla does not wrap the optimizer. It only overwrites `optimizer.lr` after every step, so it works with any optimizer and the optimizer needs no knowledge of it:

```python
optimizer.step()
scheduler.step()          # optimizer.lr = get_lr(step_count)
```

## Gradient clipping {#clipping toc="Gradient clipping"}

Now and then a batch produces a very large gradient, and a single oversized update can undo much of the progress so far. **Gradient clipping** caps the size of the update. nabla's `clip_grad_norm_` computes one norm over the gradients of all parameters, as if they were one long vector, and scales all of them by the same factor if it exceeds `max_norm`:

$$
\lVert g \rVert = \sqrt{\sum_{\text{params}} \sum_i g_i^2}, \qquad
g \leftarrow g \cdot \min\!\left(1,\; \frac{\text{max\_norm}}{\lVert g \rVert}\right)
$$

::: widget tr-04-clip

Because all gradients are scaled together, the update keeps its direction and only gets shorter. Clipping each parameter separately would cut large gradients more than small ones and change the direction the optimizer is following. Clipping belongs between `backward()` and `step()`; the function returns the norm before clipping, which is worth logging, since a norm that often hits the limit points to an unstable run.

```python
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

> [!abstract] Key points
> - Warmup then cosine decay: careful start, large steps in the middle, small steps at the end.
> - The scheduler only sets `optimizer.lr`.
> - Clip the global norm: shorter update, same direction.

## Checkpoints and logs {#checkpoints group="Long runs" toc="Checkpoints, logs"}

A run of several hours should survive a crash or a deliberate stop. `save_checkpoint` writes the model's parameters, the optimizer's state and any metadata into one file; `load_checkpoint` restores them and returns the metadata:

```python
start_step = 1
if os.path.exists(CHECKPOINT_PATH):
    metadata = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    start_step = metadata["step"] + 1
    scheduler.step_count = metadata["step"]       # keep the schedule in step

for step in range(start_step, steps + 1):
    ...
    if step % eval_interval == 0:
        save_checkpoint(CHECKPOINT_PATH, model, optimizer, step=step)
```

Both sides rely on `state_dict()` and `load_state_dict()`, defined by `Module` and `Optimizer` themselves. A model's state is a CPU copy of every parameter, in the order of `parameters()`. Adam's state is $m$, $v$ and $t$; without them a resumed run would restart Adam's averages from zero. Loading keeps every parameter on its current device, so a checkpoint saved on the CPU loads straight into a model already on the GPU. The file is a `pickle`, like PyTorch's default format, with the same caveat: only load checkpoints you trust, since unpickling can execute code.

`History` records metrics against step numbers, each metric with its own steps:

```python
history.log(step, train_loss=loss, lr=optimizer.lr, grad_norm=grad_norm)    # every step
history.log(step, val_loss=val_loss)                                        # every eval_interval steps
history.to_csv("history.csv")                                               # rows: step, metric, value
```

The CSV has one row per measurement (long format), so metrics logged at different intervals need no empty cells, and pandas or any plotting tool can pivot it as needed.

## GPU and precision {#gpu toc="GPU, float32"}

Every operation in nabla starts with `xp = get_array_module(x.data)`, which returns `numpy` for a NumPy array and `cupy` for a CuPy array on the GPU. CuPy mirrors the NumPy API, so the same code runs on either device without a single `if`. Moving a model is one call, made before the optimizer is created, so that Adam allocates its buffers on the same device:

```python
model.to("cuda")
optimizer = Adam(model.parameters(), lr=3e-4)
for X, y in loader:
    loss = criterion(model(X.to("cuda")), y)       # the loss moves the labels itself
```

At small sizes the gain is modest: for the MNIST ViT with batch size 32, a recorded measurement gave 184 ms per step on the CPU and 50 ms on the GPU. Kernel launches dominate at this size; larger models and batches benefit more.

nabla computes in `float64` by default, which keeps numerical gradient checks precise. For training, `float32` halves the memory of every activation and gradient and is much faster on a GPU:

```python
nabla.set_default_dtype(np.float32)      # before building the model
```

Only newly created parameters and constants follow the default; arrays you pass in keep their dtype, so convert input data yourself.
