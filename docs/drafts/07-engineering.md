---
title: "Engineering: GPU, precision, checkpoints, logging"
description: "CuPy backend dispatch, float32 vs. float64, checkpointing and metric tracking."
---

# Engineering: GPU, precision, checkpoints, logging

None of this post is about a new layer. It is about what it takes to move
from "the gradients are correct" to "I can train something for an hour and
trust the result": running on a GPU, controlling precision, surviving an
interrupted run and keeping track of what happened.

## GPU Support via CuPy

### The idea: one array-backend abstraction, everywhere

Every op in nabla is written against a small set of NumPy calls
(`np.exp`, `np.matmul`, `np.concatenate`, `.reshape()`, `.sum()`, ...).
[CuPy](https://cupy.dev/) implements almost the exact same API for
`cupy.ndarray` as NumPy does for `np.ndarray` — the same trick Chainer
(the library CuPy was originally built for) used: rather than branching
on device everywhere, look up *which module* a given array already
belongs to, and call functions on that module instead of hardcoding `np`.

[`backend.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/backend.py) is the one place that trick
lives:

```python
def get_array_module(*arrays):
    if CUPY_AVAILABLE:
        return cp.get_array_module(*arrays)
    return np
```

Every op's `forward`/`backward` now starts with `xp =
get_array_module(x.data)` and calls `xp.exp(...)`, `xp.sum(...)`, etc.
instead of `np.exp`/`np.sum` — `xp` resolves to `numpy` for a
CPU-backed `Tensor` and `cupy` for a GPU-backed one, and the op's logic
never needs an `if device == "cuda"` branch anywhere. CuPy is an
**optional** dependency (`pip install nabla[gpu]` /
`uv sync --extra gpu`) — `CUPY_AVAILABLE` guards the import so the
entire library still works with only NumPy installed.

### The `.to(device)` API

Mirrors PyTorch:

```python
model.to("cuda")                    # moves every parameter/buffer/submodule
x = Tensor(np_array).to("cuda")     # moves one tensor's data (and .grad, if any)
```

`Tensor.to()` ([`tensor.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/tensor.py)) calls
`backend.to_device()`, which converts via `cupy.asarray`/`cupy.asnumpy`
(a no-op if the array is already on the target device).
`Module.to()` ([`nn/module.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/module.py)) recurses
through `self.__dict__` and moves three kinds of attributes:

- **`Tensor`s** — both trainable parameters (`requires_grad=True`) and
  fixed-but-Tensor-wrapped buffers like `VisionTransformer`'s CLS-token
  selector (`requires_grad=False`, so invisible to `parameters()`, but
  it still has to move with the model).
- **plain `ndarray`/`cupy.ndarray` buffers** — fixed data that was never
  wrapped in a `Tensor` at all, like `PositionalEncoding`'s sinusoidal
  table.
- **nested `Module`s** — recursed into, same as `parameters()`/`train()`.

**Order matters**: call `model.to("cuda")` *before* constructing the
optimizer. `Adam`/`SGD` allocate their momentum buffers
(`np.zeros_like(param.data)`/`xp.zeros_like`) once, at construction
time, matching whatever device each parameter is on *right then* — the
same gotcha PyTorch has for the identical reason.

### Bugs this surfaced (all found by actually running on a GPU)

Every one of these passed the full CPU test suite unchanged, then broke
the instant a real `cupy.ndarray` hit the code path — writing this
without a GPU to test against would have shipped all four silently:

1. **Hardcoded CPU constants inside forward passes.**
   `VisionTransformer.forward` built the CLS-token batch broadcast as
   `Tensor(np.zeros(...))`, and `scaled_dot_product_attention` built its
   `1/sqrt(d_k)` scale as `Tensor(np.array(...))` — both unconditionally
   `numpy`, so multiplying/adding them against GPU-resident scores
   raised `TypeError: Unsupported type <class 'numpy.ndarray'>` (CuPy
   refuses to silently mix backends in one operation, unlike NumPy+CuPy
   broadcasting some frameworks allow). Fixed by building these with
   `get_array_module(the_gpu_tensor.data)` instead of bare `np`.

2. **User-supplied masks default to CPU.** A causal mask built with
   `np.triu(...)` (as every example does) is a plain CPU array even
   when `Q`/`K`/`V` are on the GPU. Fixed inside
   `scaled_dot_product_attention` by converting with `xp.asarray(mask)`
   before adding it to `scores`, so callers never have to remember to
   move mask arrays themselves.

3. **`Transpose.backward`'s `np.argsort(self.axes)`.** `self.axes` is a
   plain Python tuple of axis indices — not device data at all — but it
   was accidentally routed through `xp.argsort`. NumPy's `argsort`
   silently accepts a plain tuple; CuPy's does not
   (`AttributeError: 'tuple' object has no attribute 'argsort'`). Fixed
   by using plain `np.argsort` unconditionally: axis-permutation
   bookkeeping has nothing to do with which backend the *data* lives on.

4. **`Concat.backward`'s split-index computation.** Same root cause as
   (3): `tensor_sizes` is a plain Python list of ints from `.shape`
   lookups, but got passed through `xp.cumsum`, which CuPy's
   implementation rejects (`expected cupy ndarray, got list`). Fixed by
   computing the cut points with `itertools.accumulate` — plain Python,
   no array library involved, since these are static ints known before
   any device array is touched.

The pattern behind (3) and (4): **not everything that touches a tensor
op is itself device data.** Shape tuples, axis permutations, and
Python-int bookkeeping should stay plain Python/`numpy`, and only
values actually stored *as* array data need `xp` dispatch. Reaching for
`xp` reflexively on anything array-shaped was the mistake both times.

### Verifying GPU results match CPU

[`tests/test_gpu.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_gpu.py) is skipped automatically
(`pytest.mark.skipif`) on any machine without a working CUDA device, so
`uv run pytest` stays green on CPU-only machines — including CI. On a
GPU machine it checks:

- `Tensor.to()`/`Module.to()` correctly move data, gradients, and every
  kind of buffer described above.
- `Linear`, `Conv2D`, `Embedding`, and a full `TransformerBlock` (with a
  causal mask) produce **numerically identical** forward outputs and
  gradients on GPU vs. CPU, given the same weights and inputs (checked
  with `np.allclose`, not just "it ran").
- A full `VisionTransformer` trains one step end to end on the GPU
  (forward, loss, backward, optimizer step) with every parameter
  receiving a gradient.

### Does it actually help?

Benchmarked the [MNIST ViT example](06-vit.md#example-mnist-digit-classification-with-a-vision-transformer)'s training step
(`embed_dim=64, num_heads=4, hidden_dim=128, num_layers=4, batch_size=32`),
after a few warmup steps (CuPy JIT-compiles its kernels on first use -
excluding warmup would understate GPU throughput):

```
CPU: 183.8 ms/step
GPU:  49.8 ms/step
speedup: 3.7x
```

A modest speedup at this scale, not a dramatic one — this model and
batch size are small enough that per-op kernel-launch overhead eats a
real fraction of the GPU's time budget, the same reason PyTorch
recommends larger batch sizes to see GPU speedups shine. The `xp`
abstraction pays off more as model/batch size grows; this example
wasn't sized for a GPU benchmark, it was sized to train in a few
minutes on a CPU-only machine (see the [ViT example](06-vit.md#example-mnist-digit-classification-with-a-vision-transformer)).

### GPU: using it

```python
from nabla.tensor import Tensor
from nabla.nn.vit import VisionTransformer
from nabla.optim.adam import Adam

model = VisionTransformer(...)
model.to("cuda")                      # before constructing the optimizer
optimizer = Adam(model.parameters(), lr=3e-4)

for X_batch, y_batch in loader:       # DataLoader always yields CPU tensors
    X_batch = X_batch.to("cuda")      # move each batch explicitly
    logits = model(X_batch)
    loss = criterion(logits, y_batch) # plain CPU y_batch is fine - CrossEntropyLoss
    ...                                # moves it to match logits' device automatically
```

## Precision: `float32` vs. `float64`

NumPy creates `float64` arrays unless told otherwise, and so does nabla by
default. That is the right default for *testing*: a central-difference
gradient check with $\epsilon = 10^{-5}$ has about ten significant digits
to work with in double precision and almost none in single.

For *training* it is the wrong default. `float32` halves the memory of
every activation and gradient, and consumer GPUs are several times faster
in single precision. One call switches what newly created parameters use
([`backend.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/backend.py)):

```python
import numpy as np
import nabla

nabla.set_default_dtype(np.float32)   # before building the model
model = MyModel()
x = nabla.Tensor(batch.astype(np.float32))
```

The default only decides what gets *created* — parameters from `init.py`,
and Python floats or lists wrapped in a `Tensor`. Arrays that already have
a dtype are never converted behind your back; `Tensor.astype()` does that
explicitly.

### Keeping a graph in one dtype is harder than it sounds

NumPy promotes mixed operands to the wider type, so a single stray
`float64` array anywhere in the forward pass silently upgrades everything
downstream. Nothing fails; training just becomes slower. Making a
`float32` model *stay* `float32` turned up three leaks, all of the same
kind — an array created inside an op without saying which dtype it should
have:

1. **The attention scale.** `1 / sqrt(d_k)` was wrapped as a
   zero-dimensional array. Python scalars are "weak" in NumPy's promotion
   rules and adapt to the other operand; zero-dimensional *arrays* are not,
   so this one constant promoted every attention score. It is now a plain
   Python float.
2. **The causal mask.** A mask built with `np.full(..., -1e9)` is `float64`.
   It is now cast to the dtype of the scores it is added to.
3. **The embedding gradient.** `Embedding.backward` allocated its
   accumulator with `xp.zeros(shape)`, so the gradient of the embedding
   table came back as `float64` even though the forward pass was clean.

[`tests/test_dtype.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_dtype.py) builds a CNN, a text
Transformer and a Vision Transformer under `float32` and asserts that the
loss, every parameter and every gradient are still `float32` after a
backward pass and an optimizer step.

## Checkpointing

### Checkpointing: why

Every example so far runs start-to-finish in one process, in a few
minutes. That stops being true once a run takes long enough to
actually be at risk of a crash, a killed process, or just wanting to
pause it - at that point, being unable to resume means losing
everything and starting over from a freshly initialized model.
Checkpointing saves a model's (and optimizer's) state to disk so a run
can pick back up where it left off.

### Built on `state_dict()`, not model internals

[`checkpoint.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/checkpoint.py) doesn't know or care what
layers a model has, or whether the optimizer is `Adam` or `SGD` - it's a
thin wrapper around `state_dict()`/`load_state_dict()` methods defined
on `Module` and `Optimizer` themselves, the same separation of concerns
`.to(device)` already established in the [GPU section above](#gpu-support-via-cupy).

`Module.state_dict()` ([`nn/module.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/nn/module.py))
returns a CPU-resident **copy** of every parameter's values, in the
same order `parameters()` already returns them:

```python
def state_dict(self):
    return [to_device(p.data, "cpu").copy() for p in self.parameters()]

def load_state_dict(self, state):
    for param, value in zip(self.parameters(), state):
        xp = get_array_module(param.data)  # preserves param's *current* device
        param.data = xp.asarray(value)
```

Loading preserves each parameter's *current* device rather than forcing
everything back to CPU - a checkpoint saved from a GPU run loads
straight back onto the GPU if the model receiving it is already there
(via `.to("cuda")`), no separate device move needed after loading.

`Optimizer.state_dict()` defaults to `{}` (stateless optimizers like
`SGD` have nothing to save beyond `.lr`, which a
[scheduler](03-training.md#learning-rate-schedules) controls, not a checkpoint). `Adam`
overrides it to save its momentum buffers (`m`, `v`) and step counter
`t` - resuming training without this would silently restart Adam's
momentum from zero, throwing away exactly the per-parameter adaptivity
Adam is supposed to provide.

### The two functions

```python
save_checkpoint(path, model, optimizer=None, **metadata)
metadata = load_checkpoint(path, model, optimizer=None)
```

`**metadata` is arbitrary extra state the training loop wants
round-tripped alongside the weights - typically at least `step` (or
`epoch`), so the training loop knows where to resume counting from, but
anything JSON/pickle-able works (e.g. `best_val_loss` to resume "only
save if better than..." logic correctly too). `load_checkpoint` returns
this dict back to the caller to do with as it likes -
[`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py)
uses it like this:

```python
start_step = 1
if os.path.exists(CHECKPOINT_PATH):
    metadata = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    start_step = metadata["step"] + 1
    scheduler.step_count = metadata["step"]  # keep the LR schedule in sync too

for step in range(start_step, steps + 1):
    ...
    if step % eval_interval == 0:
        save_checkpoint(CHECKPOINT_PATH, model, optimizer, step=step, train_loss=...)
```

Re-running the script picks up training from the last saved step
instead of starting over - including the learning-rate schedule, which
needs its own `step_count` restored too, or warmup/decay would restart
from scratch and no longer line up with the actual step number.

### Format: `pickle`, and the caveat that comes with it

Model state is a Python list of arrays and optimizer state is a small
dict - simple enough that `pickle` (the same thing PyTorch's own
`torch.save`/`torch.load` use by default) is the simplest way to
serialize the whole payload in one file, metadata included, without
inventing a custom format. The same caveat PyTorch's docs carry applies
here too: **only load checkpoints you trust** - unpickling a maliciously
crafted file can execute arbitrary code.

#### Testing

[`tests/test_checkpoint.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_checkpoint.py) checks that
`state_dict()`/`load_state_dict()` round-trip a `Linear` layer's weights
correctly and that `state_dict()` returns independent copies (mutating
the live model afterward doesn't change an already-taken snapshot), that
a full `save_checkpoint`/`load_checkpoint` round-trip restores model
weights, Adam's momentum buffers and step counter, and arbitrary
metadata, and that loading without passing an optimizer is valid (just
skips restoring optimizer state).

## Logging / Metric Tracking

### Why not just `print()`

Every example so far has printed loss values straight to the console -
fine for watching a run live, but nothing plottable survives once the
terminal scrolls past it or the process ends. [`logging.py`](https://github.com/maaaax1611/nabla/blob/main/src/nabla/logging.py)'s
`History` is deliberately the smallest thing that fixes that: an
in-memory record of named metrics against step numbers, exportable to
CSV - not a dependency on TensorBoard/wandb/etc., which would be a lot
of surface area for what a training loop this size actually needs.

### Different metrics, different steps

A training loop typically doesn't log everything at the same cadence -
train loss every step, but validation loss only every `eval_interval`
steps (recomputing val loss every step would be wasteful). `History`
handles this by keeping each metric's own list of `(step, value)`
pairs rather than one row per step with columns for every metric:

```python
def log(self, step, **metrics):
    for name, value in metrics.items():
        self.records.setdefault(name, []).append((step, float(value)))
```

`history.log(step, train_loss=..., lr=..., grad_norm=...)` logs three
metrics at once at the same step;
`history.log(step, val_loss=...)` a few lines later, only every
`eval_interval` steps, logs a fourth metric that simply has fewer
recorded points than the others - no padding or `NaN`-filling needed
for the steps it wasn't logged at.

### Export: long/tidy CSV, not wide

`to_csv` writes one row per `(step, metric, value)` triple:

```
step,metric,value
1,train_loss,5.14
1,lr,6e-05
5,train_loss,4.54
5,val_loss,4.36
```

rather than one column per metric - a wide format (`step, train_loss,
val_loss, lr, ...`) would need every row to have a value for every
column, which doesn't fit metrics logged at different steps without
inventing a placeholder for the gaps. The long format sidesteps that
entirely, and is what most plotting tools (pandas `pivot`, seaborn,
etc.) actually want as input anyway.

### Used in the Shakespeare example

[`examples/shakespeare_transformer.py`](https://github.com/maaaax1611/nabla/blob/main/examples/shakespeare_transformer.py)
logs train loss, learning rate ([schedules](03-training.md#learning-rate-schedules)), and
gradient norm ([clipping](03-training.md#gradient-clipping)) every step, and val
loss every `eval_interval` steps, then writes the whole run to
`.shakespeare_history.csv` at the end - one file with everything needed
to plot the full training curve alongside the LR schedule and gradient
norm over time, without re-running training.

#### Testing

[`tests/test_logging.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_logging.py) checks logging and
retrieving a single metric, logging multiple metrics at once, that
metrics logged at different steps stay independent, `last()`'s
most-recent-value lookup (and its `None` result for a metric never
logged), and that `to_csv` produces the expected long-format rows.

---

Previous: [Vision Transformer](06-vit.md)
