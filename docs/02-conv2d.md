# Conv2D

## The idea

A 2D convolution slides a small kernel over an image and, at every
position, computes a dot product between the kernel and the patch of the
image underneath it. Stacking multiple kernels (`out_channels` of them)
gives multiple output feature maps at once.

Strictly speaking, what deep learning frameworks call "convolution" is
*cross-correlation* (the kernel is not flipped before sliding it across the
input) — nabla follows that same convention, since it makes no practical
difference for a kernel whose weights are learned anyway.

## The math

For input $x$ of shape `(batch, in_channels, H, W)`, weight $w$ of shape
`(out_channels, in_channels, kh, kw)` and bias $b$ of shape
`(out_channels,)`:

$$
\text{out}[n, o, i, j] = b[o] + \sum_{c=1}^{C} \sum_{p=1}^{kh} \sum_{q=1}^{kw} x[n, c, i \cdot s + p,\ j \cdot s + q] \cdot w[o, c, p, q]
$$

where $s$ is the stride, and the output spatial size is

$$
\text{out}_h = \left\lfloor \frac{H + 2 \cdot \text{padding} - kh}{s} \right\rfloor + 1 \qquad (\text{analogous for } \text{out}_w)
$$

## im2col: turning convolution into one matrix multiply

Computing that triple sum with Python loops over every output position
would be slow. The standard trick — **im2col** ("image to columns") — turns
the whole convolution into a single matrix multiply instead:

1. For every output position `(i, j)`, extract the `(in_channels, kh, kw)`
   patch of the input that the kernel would slide over there.
2. Flatten each patch into a column of length `in_channels * kh * kw`.
   Stack all `batch * out_h * out_w` of these columns side by side into a
   matrix `cols` of shape `(in_channels*kh*kw, batch*out_h*out_w)`.
3. Flatten the weight into `(out_channels, in_channels*kh*kw)`.
4. The whole convolution is now one matmul: `weight_flat @ cols`, shape
   `(out_channels, batch*out_h*out_w)`.
5. Reshape that back into `(batch, out_channels, out_h, out_w)`.

## Implementation walkthrough

[`ops/conv.py`](../src/nabla/ops/conv.py) implements exactly those five
steps:

```python
windows = sliding_window_view(self.x_padded, (kh, kw), axis=(2, 3))
windows = windows[:, :, ::self.stride, ::self.stride, :, :]
self.cols = windows.transpose(1, 4, 5, 0, 2, 3).reshape(in_channels * kh * kw, -1)
```

`sliding_window_view` is what makes step 1 fast: instead of copying every
patch, it returns a *view* over the original array with an extra `(kh, kw)`
window dimension — no data is duplicated, NumPy just computes different
strides into the same memory. Padding is applied once upfront
(`np.pad`) so the padded region can be sliced like any other pixel.

```python
weight_flat = weight.data.reshape(weight.data.shape[0], -1)
out_flat = weight_flat @ self.cols + bias.data.reshape(-1, 1)
out = out_flat.reshape(weight.data.shape[0], self.batch_size, self.out_h, self.out_w)
return out.transpose(1, 0, 2, 3)
```

> **Bug we hit:** the last line looks like it should be equivalent to
> `out_flat.reshape(batch_size, out_channels, out_h, out_w)` directly — it
> is not. `reshape` only *reinterprets* the existing flat memory layout; it
> has no idea which axis is "supposed to" move where. `out_flat`'s memory is
> laid out as `(out_channels, batch*out_h*out_w)`, so reshaping straight
> into `(batch, out_channels, out_h, out_w)` silently scrambles values
> whenever `batch > 1` *and* `out_channels > 1` (with either one equal to 1
> the bug is invisible, which is exactly how it slipped through at first).
> The fix is to reshape into the shape that matches the *current* memory
> order first (`(out_channels, batch, out_h, out_w)`), and only then
> `transpose` — transpose actually reorders axes, reshape does not.

## Backward

Because forward is just "reshape → matmul → reshape", backward is the same
matmul-based reasoning in reverse, computed once `cols` and `weight_flat`
are around from forward:

- **`grad_bias`**: the bias was simply added to every output position, so
  its gradient is the sum of `grad_output` over every axis except
  `out_channels`: `sum(grad_output, axis=(0, 2, 3))`.
- **`grad_weight`**: `weight_flat @ cols` is a plain matmul, so
  `grad_weight_flat = grad_output_flat @ cols.T` (standard matmul backward,
  see [Autodiff basics](01-autodiff.md#the-idea) — `MatMul.backward` in
  [`ops/transform.py`](../src/nabla/ops/transform.py) does the same thing).
- **`grad_x`**: first compute `grad_cols = weight_flat.T @ grad_output_flat`
  — the gradient w.r.t. the *extracted patches*. Then that has to be
  scattered back into the padded input at the same positions the patches
  were extracted from in forward, accumulating (`+=`) wherever windows
  overlapped (i.e. `stride < kernel_size`):

  ```python
  for i in range(self.out_h):
      for j in range(self.out_w):
          patch_grad = grad_cols[:, :, :, :, i, j].transpose(3, 0, 1, 2)
          h0, w0 = i * self.stride, j * self.stride
          grad_x_padded[:, :, h0:h0 + kh, w0:w0 + kw] += patch_grad
  ```

  Finally, the padding that was added in forward gets sliced back off.

All of this is checked against a hand-written triple-loop reference
implementation and against numerical (finite-difference) gradients in
[`tests/test_conv_ops.py`](../tests/test_conv_ops.py).
