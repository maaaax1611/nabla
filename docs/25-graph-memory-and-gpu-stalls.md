# Graph Memory and GPU Stalls

`src/nabla/tensor.py` (`Tensor.backward()`)

This one didn't start as a memory-management investigation - it started as
"scale the Shakespeare model up 30x and see what happens." What happened
was a training loop that ran at wildly inconsistent speeds, from under a
second to over 40 seconds per step, with no code difference between steps.

## The symptom

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

## Chasing it down

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

## The result

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

## Why the small model never showed this

Both cycles existed the whole time - they're not new bugs introduced by
scaling up. What changed is how much memory got trapped in each one
before Python's cyclic collector happened to run. At 158K parameters, a
few un-freed graphs are a rounding error next to an 8 GB GPU; the
allocator never had to work hard to satisfy a new request, so nothing
ever stalled long enough to notice. At 4.8M parameters (and longer
sequences, and a bigger batch), a handful of un-freed graphs was already
gigabytes, so the same collection-timing behavior turned into
multi-second, once-every-few-steps stalls. This is the same shape of
lesson as the [GPU dispatch bugs](19-gpu-support.md): correctness bugs
that a small enough test case simply won't reveal.

## Testing

[`tests/test_graph_lifecycle.py`](../tests/test_graph_lifecycle.py) checks
that `_ctx`/`_prev` are cleared on both intermediate and root tensors
after `backward()`, that leaf tensors are untouched and remain reusable
in a fresh graph afterward, that clearing the graph doesn't disturb the
gradients that were computed from it, and - with `gc` disabled so only
plain refcounting can act - that a `matmul` + `sum` graph leaves zero
objects for the cyclic collector to find at all.
