# Engineering report

Problems that came up while building nabla, what caused them and how they
were fixed. The documentation in `docs/` describes how the library works;
this file records how it got there.

| # | Area | Symptom | Cause |
| --- | --- | --- | --- |
| 1 | Broadcasting | gradient in the wrong shape, crash in the next op | `unbroadcast` only summed over added leading axes, not over stretched size-1 axes |
| 2 | Conv2D | scrambled output for batch > 1 and several output channels | `reshape` used where the memory layout needed a `transpose` |
| 3 | Max pooling | gradient shaped (batch, channels, batch, channels) | slices mixed with array indices in the scatter |
| 4 | BatchNorm2D | `AttributeError` on the first `backward()` | a value only backward needs was never cached in forward |
| 5 | Concat | gradients assigned to the wrong inputs | extra leading 0 in the split indices |
| 6 | GPU | four device mismatches on the first real GPU run | CPU arrays created inside ops and modules |
| 7 | Memory | multi-second stalls per training step on GPU | two reference cycles kept every step's graph alive |
| 8 | dtype | float32 models silently computed in float64 | arrays created without an explicit dtype |

## 1. Broadcasting: `unbroadcast` missed stretched axes

The first version of `unbroadcast` only collapsed *extra
leading* dimensions (the `while` loop) — the case where `bias` has fewer
dimensions than `x`, like `(C,)` vs `(N, C)`. It missed the case where both
shapes have the *same* number of dimensions but a size-1 axis got
broadcast up, like `(1, C, 1, 1)` vs `(N, C, H, W)` — which is exactly
what BatchNorm2D does in eval mode. Gradients came back
in the wrong shape and crashed the next `Function.backward` downstream.
A second loop (summing any axis where `target_dim == 1` but
`grad_dim != 1`) fixes the general case.

## 2. Conv2D: `reshape` is not `transpose`

`Conv2D.forward` computes the convolution as one matrix product, `out_flat`,
and then has to bring it into `(batch, out_channels, out_h, out_w)`. That last step looks like it should be equivalent to
`out_flat.reshape(batch_size, out_channels, out_h, out_w)` directly — it
is not. `reshape` only *reinterprets* the existing flat memory layout; it
has no idea which axis is "supposed to" move where. `out_flat`'s memory is
laid out as `(out_channels, batch*out_h*out_w)`, so reshaping straight
into `(batch, out_channels, out_h, out_w)` silently scrambles values
whenever `batch > 1` *and* `out_channels > 1` (with either one equal to 1
the bug is invisible, which is exactly how it slipped through at first).
The fix is to reshape into the shape that matches the *current* memory
order first (`(out_channels, batch, out_h, out_w)`), and only then
`transpose` — transpose actually reorders axes, reshape does not.

## 3. Max pooling: slices mixed with array indices

`MaxPool2D.backward` scatters each gradient value back to the position of
its window's maximum. The first version indexed with
`grad_x[:, :, h_idx, w_idx]` — mixing plain slices (`:`) for the batch and
channel axes with *array* indices (`h_idx`, `w_idx`, each shape
`(batch, channels)`) for the spatial axes. NumPy's advanced-indexing rules
don't broadcast that the way you'd hope: since the array indices aren't
separated by a slice, they get inserted as their *own* dimensions at that
position — the result ends up shaped `(batch, channels, batch, channels)`,
effectively an outer product across every batch/channel pair instead of
the intended one-to-one match per sample. The fix is to make the batch and
channel axes into broadcastable arrays too
(`np.arange(batch)[:, None]`, `np.arange(channels)[None, :]`), so *all
four* index arrays broadcast together elementwise and each `(b, c)`
position only ever touches its own `(h, w)`.

## 4. BatchNorm2D: value for backward not cached

`forward` originally never set `self.N`
(`batch * H * W`, the count backward divides by in steps 4 and 5 above) —
an easy thing to forget since it's not needed anywhere in forward itself,
only in backward. First call to `backward()` raised
`AttributeError: 'BatchNorm2D' object has no attribute 'N'`. A reminder
that "what does forward need to *cache* for backward" is a separate
question from "what does forward need to *compute* its own output".

## 5. Concat: off-by-one in the backward pass

An earlier version computed `split_indices` as
`cumsum([0] + tensor_sizes)[:-1]`. Prepending that `0` shifts every cut
point down by one slot and, worse, adds a phantom leading split: for
three tensors, `np.split` returns *four* pieces (`[0:0]`, `[0:3]`,
`[3:8]`, `[8:10]`) — an empty array first, then everything after it
misaligned with the wrong input. Numerical gradient checking on all
three inputs caught it immediately. The fix was just to drop the
leading `0` — `cumsum(tensor_sizes)[:-1]` already starts at the first
real cut point.

## 6. GPU: bugs that only a real device showed

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

## 7. Graph memory and GPU stalls

`src/nabla/tensor.py` (`Tensor.backward()`)

This one didn't start as a memory-management investigation - it started as
"scale the Shakespeare model up 30x and see what happens." What happened
was a training loop that ran at wildly inconsistent speeds, from under a
second to over 40 seconds per step, with no code difference between steps.

### The symptom

Scaling `examples/shakespeare_transformer.py` from ~158K parameters
(embed_dim=64, 3 layers) to ~4.8M parameters (embed_dim=256, 6 layers,
longer sequences) made per-step time on the GPU completely erratic:

```
step 0: 806.5 ms   | pool used 4128.0 MB
step 1: 1438.0 ms  | pool used 8103.1 MB
step 2: 4150.5 ms  | pool used 12078.2 MB
step 3: 4525.5 ms  | pool used 16053.2 MB
step 4: 8409.0 ms  | pool used 4128.0 MB   <- drops back down
step 5: 2988.5 ms  | pool used 8103.1 MB
...
step 9: 13059.0 ms | pool used 4128.0 MB
```

CuPy's memory pool (`cupy.get_default_memory_pool().used_bytes()`) grew by
roughly one step's worth of activations *every single step*, for several
steps in a row, before suddenly dropping back down - and the steps where
it dropped were the slowest ones by far. The smaller model never showed
this pattern at all; its memory usage was flat step to step.

### Chasing it down

The growing-then-dropping pattern is the signature of relying on Python's
*cyclic* garbage collector instead of plain reference counting. CPython
frees an object the instant its refcount hits zero - but if two objects
reference each other (directly or through a longer chain), their
refcounts never reach zero on their own, no matter how unreachable the
whole cycle is from the rest of the program. Only the generational
collector's cycle-finder can free those, and it doesn't run on every
allocation - it runs based on an internal allocation-count heuristic. So
memory tied up in cycles keeps accumulating until that collector happens
to fire, at which point everything gets freed in one lump - which is
exactly what step 4's 8.4-second stall was: CuPy's allocator scrambling to
service new allocations while GPU memory was still full of the previous
several steps' un-freed graphs, right up until gc finally ran mid-step.

Confirming it: manually calling `gc.collect()` after `backward()`
(instead of waiting on Python to decide) immediately dropped the pool from
4.1 GB to 153 MB and reported reclaiming hundreds of objects - proof the
"leaked" memory was cyclic garbage, not a real leak.

`gc.get_referrers()` on the collected objects pointed at two independent
cycles:

**1. The autodiff graph itself.** Every non-leaf `Tensor` holds a
`Function` via `_ctx`; every `Function` holds its input tensors via
`saved_tensors`. That's not a cycle by itself (children point at parents,
parents never point forward at children) - but once a full graph has been
built and walked, nothing needs those links anymore. `Tensor.backward()`
now clears `_ctx` and `_prev` on every tensor in the graph right after
computing gradients, the same "free the graph by default" behavior
PyTorch has when you don't pass `retain_graph=True`.

**2. `topo_sort`'s own closure.** The topological sort inside
`backward()` used to be a recursive nested function:

```python
def topo_sort(tensor: Tensor) -> None:
    if id(tensor) not in visited:
        visited.add(id(tensor))
        for parent in tensor._prev:
            topo_sort(parent)   # <- refers to itself
        topo.append(tensor)

topo_sort(self)
```

A nested function that calls itself by name resolves that name through
its own closure cell - which means the function object holds a reference
to a cell that (once the recursive call has happened) refers back to the
function object itself. That's a reference cycle completely independent
of anything tensor-related, and it kept `backward()`'s entire stack frame
- including the `topo` list, which by construction holds a reference to
*every tensor in the graph* - alive until the cyclic collector ran. This
one mattered more than it looks: even after fix #1 cleared every tensor's
own `_ctx`/`_prev`, the `topo` list still held direct references to all
of them, so the graph stayed alive regardless.

The fix is to make the traversal iterative instead - an explicit stack
carrying `(tensor, expanded)` pairs reproduces the same post-order a
recursive DFS would, with no closure and therefore no self-reference:

```python
topo: list[Tensor] = []
visited: set[int] = set()
stack: list[tuple[Tensor, bool]] = [(self, False)]
while stack:
    tensor, expanded = stack.pop()
    if expanded:
        topo.append(tensor)
    elif id(tensor) not in visited:
        visited.add(id(tensor))
        stack.append((tensor, True))
        for parent in tensor._prev:
            stack.append((parent, False))
```

### The result

With both cycles gone, the 4.8M-parameter model's per-step time on the
GPU became flat and predictable - no `gc.collect()` workaround needed in
the training script itself, because there's no cyclic garbage left for it
to have to clean up:

```
step 0: 910.0 ms | pool used 153.0 MB
step 1: 701.0 ms | pool used 153.0 MB
step 2: 700.0 ms | pool used 153.0 MB
...
step 14: 709.0 ms | pool used 153.0 MB
```

### Why the small model never showed this

Both cycles existed the whole time - they're not new bugs introduced by
scaling up. What changed is how much memory got trapped in each one
before Python's cyclic collector happened to run. At 158K parameters, a
few un-freed graphs are a rounding error next to an 8 GB GPU; the
allocator never had to work hard to satisfy a new request, so nothing
ever stalled long enough to notice. At 4.8M parameters (and longer
sequences, and a bigger batch), a handful of un-freed graphs was already
gigabytes, so the same collection-timing behavior turned into
multi-second, once-every-few-steps stalls. This is the same shape of
lesson as the GPU dispatch bugs: correctness bugs
that a small enough test case simply won't reveal.

### Testing

[`tests/test_graph_lifecycle.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_graph_lifecycle.py) checks
that `_ctx`/`_prev` are cleared on both intermediate and root tensors
after `backward()`, that leaf tensors are untouched and remain reusable
in a fresh graph afterward, that clearing the graph doesn't disturb the
gradients that were computed from it, and - with `gc` disabled so only
plain refcounting can act - that a `matmul` + `sum` graph leaves zero
objects for the cyclic collector to find at all.

## 8. dtype: float32 graphs promoted to float64

`set_default_dtype(np.float32)` creates float32 parameters, but NumPy
promotes mixed operands to the wider type, so one stray float64 array in
the forward pass upgrades everything downstream. Nothing fails; training
is only slower. Three places created arrays without saying which dtype
they should have:

1. **Attention scale.** `1 / sqrt(d_k)` was wrapped as a zero-dimensional
   array. Python scalars adapt to the other operand under NumPy's
   promotion rules, zero-dimensional arrays do not. It is now a plain
   Python float.
2. **Causal mask.** A mask built with `np.full(..., -1e9)` is float64. It
   is now cast to the dtype of the scores it is added to.
3. **Embedding gradient.** `Embedding.backward` allocated its accumulator
   with `xp.zeros(shape)`, so the gradient of the table came back as
   float64 although the forward pass was clean.

[`tests/test_dtype.py`](https://github.com/maaaax1611/nabla/blob/main/tests/test_dtype.py)
builds a CNN, a text Transformer and a Vision Transformer under float32
and asserts that loss, parameters and gradients are still float32 after a
backward pass and an optimizer step.
