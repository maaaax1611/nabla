# Pooling

## The idea

Pooling shrinks a feature map spatially by summarizing each small window
with a single number — either its maximum or its average. Unlike
[Conv2D](02-conv2d.md), there are no learnable weights: pooling is a fixed
function of its input, and it doesn't mix channels (each channel is pooled
independently).

Both `MaxPool2D` and `AvgPool2D`
([`ops/pooling.py`](../src/nabla/ops/pooling.py)) share the same forward
skeleton — extract windows with `sliding_window_view`, take every
`stride`-th one, flatten the `(kh, kw)` window into one axis, and reduce
over it:

```python
windows = sliding_window_view(x.data, (kernel_size, kernel_size), axis=(2, 3))
windows = windows[:, :, ::stride, ::stride, :, :]
windows = windows.reshape(*windows.shape[:4], -1)  # (batch, channels, out_h, out_w, kh*kw)
```

Where they differ is the reduction — `np.max` vs. `np.mean` — and, much
more interestingly, in what that means for the gradient.

## Max pooling: the gradient is a router, not a formula

$$
\text{out}[n, c, i, j] = \max_{(p, q) \in \text{window}(i, j)} x[n, c, p, q]
$$

`max` is not smooth everywhere, but it *is* differentiable almost
everywhere, and where it is:

$$
\frac{\partial \, \text{out}[n,c,i,j]}{\partial \, x[n,c,p,q]} =
\begin{cases}
1 & \text{if } x[n,c,p,q] \text{ is the maximum in the window} \\
0 & \text{otherwise}
\end{cases}
$$

So the backward pass doesn't need any arithmetic on the incoming gradient —
it just needs to know *which* input position was the max for each output
position, and route the entire gradient there unchanged. `forward` records
exactly that with `np.argmax`:

```python
self.argmax = np.argmax(windows, axis=-1)  # (batch, channels, out_h, out_w)
```

and `backward` scatters `grad_output` back to those positions,
**accumulating** wherever overlapping windows (`stride < kernel_size`) route
gradient to the same input pixel more than once:

```python
for i in range(self.out_h):
    for j in range(self.out_w):
        di, dj = np.unravel_index(self.argmax[:, :, i, j], (kernel_size, kernel_size))
        h_idx, w_idx = i * stride + di, j * stride + dj
        np.add.at(grad_x, (batch_idx, channel_idx, h_idx, w_idx), grad_output[:, :, i, j])
```

> **Bug we hit:** the first version of this indexed with
> `grad_x[:, :, h_idx, w_idx]` — mixing plain slices (`:`) for the batch and
> channel axes with *array* indices (`h_idx`, `w_idx`, each shape
> `(batch, channels)`) for the spatial axes. NumPy's advanced-indexing rules
> don't broadcast that the way you'd hope: since the array indices aren't
> separated by a slice, they get inserted as their *own* dimensions at that
> position — the result ends up shaped `(batch, channels, batch, channels)`,
> effectively an outer product across every batch/channel pair instead of
> the intended one-to-one match per sample. The fix is to make the batch and
> channel axes into broadcastable arrays too
> (`np.arange(batch)[:, None]`, `np.arange(channels)[None, :]`), so *all
> four* index arrays broadcast together elementwise and each `(b, c)`
> position only ever touches its own `(h, w)`.

Note also what happens on ties: `np.argmax` always returns the *first*
occurrence of the maximum. That's an arbitrary but standard convention —
the subgradient of `max` is technically any convex combination of the tied
inputs, and frameworks universally just pick one.

## Average pooling: a genuinely linear op

$$
\text{out}[n, c, i, j] = \frac{1}{kh \cdot kw} \sum_{(p, q) \in \text{window}(i,j)} x[n, c, p, q]
$$

Average pooling is linear in $x$, so unlike max pooling its gradient
doesn't depend on the input values at all — every position in the window
contributed equally to the output, so every position gets an equal share of
the gradient back:

$$
\frac{\partial \, \text{out}[n,c,i,j]}{\partial \, x[n,c,p,q]} = \frac{1}{kh \cdot kw} \quad \text{for every } (p,q) \text{ in the window}
$$

which is exactly what `backward` does — no argmax bookkeeping needed, just
spread `grad_output / (kh*kw)` over the whole window (again accumulating on
overlap):

```python
grad_per_position = grad_output[:, :, i, j] / (kernel_size * kernel_size)
grad_x[:, :, h0:h0+kernel_size, w0:w0+kernel_size] += grad_per_position[:, :, None, None]
```

Both ops are checked against triple-loop reference implementations and
numerical gradients — including overlapping-window cases (`stride <
kernel_size`) — in
[`tests/test_pooling_ops.py`](../tests/test_pooling_ops.py).
