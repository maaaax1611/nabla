# Checkpointing

## Why

Every example so far runs start-to-finish in one process, in a few
minutes. That stops being true once a run takes long enough to
actually be at risk of a crash, a killed process, or just wanting to
pause it - at that point, being unable to resume means losing
everything and starting over from a freshly initialized model.
Checkpointing saves a model's (and optimizer's) state to disk so a run
can pick back up where it left off.

## Built on `state_dict()`, not model internals

[`checkpoint.py`](../src/nabla/checkpoint.py) doesn't know or care what
layers a model has, or whether the optimizer is `Adam` or `SGD` - it's a
thin wrapper around `state_dict()`/`load_state_dict()` methods defined
on `Module` and `Optimizer` themselves, the same separation of concerns
`.to(device)` already established in [docs/19](19-gpu-support.md).

`Module.state_dict()` ([`nn/module.py`](../src/nabla/nn/module.py))
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
[scheduler](20-lr-schedules.md) controls, not a checkpoint). `Adam`
overrides it to save its momentum buffers (`m`, `v`) and step counter
`t` - resuming training without this would silently restart Adam's
momentum from zero, throwing away exactly the per-parameter adaptivity
Adam is supposed to provide.

## The two functions

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
[`examples/shakespeare_transformer.py`](../examples/shakespeare_transformer.py)
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

## Format: `pickle`, and the caveat that comes with it

Model state is a Python list of arrays and optimizer state is a small
dict - simple enough that `pickle` (the same thing PyTorch's own
`torch.save`/`torch.load` use by default) is the simplest way to
serialize the whole payload in one file, metadata included, without
inventing a custom format. The same caveat PyTorch's docs carry applies
here too: **only load checkpoints you trust** - unpickling a maliciously
crafted file can execute arbitrary code.

## Testing

[`tests/test_checkpoint.py`](../tests/test_checkpoint.py) checks that
`state_dict()`/`load_state_dict()` round-trip a `Linear` layer's weights
correctly and that `state_dict()` returns independent copies (mutating
the live model afterward doesn't change an already-taken snapshot), that
a full `save_checkpoint`/`load_checkpoint` round-trip restores model
weights, Adam's momentum buffers and step counter, and arbitrary
metadata, and that loading without passing an optimizer is valid (just
skips restoring optimizer state).
