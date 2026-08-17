# GPU Support via CuPy

## The idea: one array-backend abstraction, everywhere

Every op in nabla is written against a small set of NumPy calls
(`np.exp`, `np.matmul`, `np.concatenate`, `.reshape()`, `.sum()`, ...).
[CuPy](https://cupy.dev/) implements almost the exact same API for
`cupy.ndarray` as NumPy does for `np.ndarray` — the same trick Chainer
(the library CuPy was originally built for) used: rather than branching
on device everywhere, look up *which module* a given array already
belongs to, and call functions on that module instead of hardcoding `np`.

[`backend.py`](../src/nabla/backend.py) is the one place that trick
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

## The `.to(device)` API

Mirrors PyTorch:

```python
model.to("cuda")                    # moves every parameter/buffer/submodule
x = Tensor(np_array).to("cuda")     # moves one tensor's data (and .grad, if any)
```

`Tensor.to()` ([`tensor.py`](../src/nabla/tensor.py)) calls
`backend.to_device()`, which converts via `cupy.asarray`/`cupy.asnumpy`
(a no-op if the array is already on the target device).
`Module.to()` ([`nn/module.py`](../src/nabla/nn/module.py)) recurses
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

## Bugs this surfaced (all found by actually running on a GPU)

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

## Verifying GPU results match CPU

[`tests/test_gpu.py`](../tests/test_gpu.py) is skipped automatically
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

## Does it actually help?

Benchmarked the [MNIST ViT example](18-vit-example.md)'s training step
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
minutes on a CPU-only machine (see [docs/18](18-vit-example.md)).

## Using it

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
