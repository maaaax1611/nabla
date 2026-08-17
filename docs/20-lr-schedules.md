# Learning-Rate Schedules

## Why a schedule, not a fixed `lr`

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

## The design: a scheduler mutates `optimizer.lr`, nothing else

[`optim/scheduler.py`](../src/nabla/optim/scheduler.py)'s `LRScheduler`
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

## `WarmupCosineLR`

The schedule used in [`examples/shakespeare_transformer.py`](../examples/shakespeare_transformer.py):
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

## `StepLR`

A simpler, more traditional schedule: multiply `lr` by `gamma` every
`step_size` steps. Less common for Transformers, but a standard choice
for CNNs (`examples/mnist_cnn.py` doesn't use it yet, but could):

```python
def get_lr(self, step):
    return self.base_lr * (self.gamma ** (step // self.step_size))
```

## Testing

[`tests/test_scheduler.py`](../tests/test_scheduler.py) checks that
warmup linearly reaches `base_lr` at exactly `warmup_steps`, that cosine
decay reaches `min_lr` at exactly `total_steps` and stays there
afterward, that the schedule is monotonically non-increasing after
warmup ends, that `StepLR` decays by the right power of `gamma` at each
boundary, and that constructing a scheduler with `warmup_steps >
total_steps` is rejected outright rather than producing a nonsensical
schedule.
