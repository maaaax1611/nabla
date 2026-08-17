# Gradient Clipping

## The problem it guards against

Deep stacks of layers - many stacked `TransformerBlock`s, for instance -
can occasionally produce a gradient that's enormous for one batch (a bad
batch, an unlucky combination of activations, early in training before
things have settled). Taking an optimizer step with that gradient can
throw the model's weights somewhere it never recovers from - "exploding
gradients," the same failure mode residual connections
([docs/13](13-transformer-block.md)) help prevent on the *forward* pass,
but nothing so far protects against on the *backward* pass.

## Clipping the combined norm, not each gradient independently

[`optim/clip.py`](../src/nabla/optim/clip.py)'s `clip_grad_norm_`
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

## Where it goes in the training loop

Between `backward()` and `optimizer.step()` - clipping only makes sense
once every gradient has been computed, and before they get used to
update anything:

```python
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The returned `total_norm` (the norm *before* clipping) is worth logging
on its own - see [`examples/shakespeare_transformer.py`](../examples/shakespeare_transformer.py),
which logs it every step via [`History`](23-logging.md). A norm that's
frequently at or near `max_norm` is a useful signal that training is
running hot, even without ever looking at the loss curve.

## Testing

[`tests/test_clip.py`](../tests/test_clip.py) checks the returned norm
against a hand-computed value, that gradients are left untouched when
already under `max_norm`, that clipping preserves direction (the
clipped gradient is a positive scalar multiple of the original), that
the norm is combined correctly across *multiple* parameters (not summed
as if each had its own independent budget), that every parameter gets
scaled by the *same* factor, and that parameters with no gradient are
skipped without error.
