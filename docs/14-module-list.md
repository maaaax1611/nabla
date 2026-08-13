# ModuleList

## The problem it solves

Stacking several of the same layer — several
[`TransformerBlock`s](13-transformer-block.md), for instance — reads
naturally as a Python list:

```python
self.blocks = [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
```

But [`Module.parameters()`/`train()`/`eval()`](01-autodiff.md) only look at
attributes that are themselves a `Tensor` or a `Module` — a plain `list`
attribute is neither, so every parameter inside those blocks would be
silently invisible to `parameters()` (no gradient updates would ever reach
them) and `train()`/`eval()` would never propagate into them either. This
is exactly the gap PyTorch's `nn.ModuleList` exists to close — a plain list
doesn't work there either, for the identical reason.

## How `ModuleList` closes it

[`nn/container.py`](../src/nabla/nn/container.py) doesn't add any new
logic to `Module.parameters()`/`train()`/`eval()` at all — it just stores
each submodule under its own generated attribute name, so the *existing*
introspection (which already recurses into any `Module`-typed attribute)
picks every one of them up automatically:

```python
def __init__(self, modules: Iterable[Module]) -> None:
    super().__init__()
    self._items: list[Module] = list(modules)
    for i, module in enumerate(self._items):
        setattr(self, f"_module_{i}", module)
```

`self._items` itself stays a plain list (harmless — it's not a `Tensor` or
`Module`, so `parameters()` skips right over it), used only for Python-side
indexing and iteration (`__getitem__`, `__iter__`, `__len__`), while
`self._module_0`, `self._module_1`, ... are what actually make every
submodule's parameters and train/eval state reachable.

## Testing

[`tests/test_nn_container.py`](../tests/test_nn_container.py) checks
indexing and iteration order, that `parameters()` returns every submodule's
parameters (not just the first, or none), that `train()`/`eval()`
propagates to every item, and that gradients flow correctly through a small
stack of layers end to end.
