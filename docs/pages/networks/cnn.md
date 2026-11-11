---
title: Convolutional networks
desc: Convolution as a sliding dot product and as one matrix product (im2col), pooling, batch normalization, and a small CNN for MNIST.
tags: [Conv2D, im2col, Pooling, BatchNorm, MNIST]
order: 2
eyebrow: Networks · Images
footer: Code excerpts are from <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/conv.py"><code>ops/conv.py</code></a>, <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/pooling.py"><code>ops/pooling.py</code></a> and <a href="https://github.com/maaaax1611/nabla/blob/main/src/nabla/ops/batchnorm.py"><code>ops/batchnorm.py</code></a>, shortened to the essentials. The example is <a href="https://github.com/maaaax1611/nabla/blob/main/examples/mnist_cnn.py"><code>examples/mnist_cnn.py</code></a>.
---

A linear layer treats an image as a flat list of numbers and has to learn from scratch that neighbouring pixels belong together. A convolution builds this in: a small filter looks at one neighbourhood at a time, and the same filter is used everywhere in the image. This page covers the three operations a classic convolutional network is made of, convolution, pooling and batch normalization, with their backward passes, and ends with a small network for handwritten digits.

## Why convolution {#why group="Convolution" toc="Why convolution"}

Two properties of images make a full linear layer wasteful:

- **Locality.** Whether a pixel belongs to an edge depends on its neighbours, not on pixels at the other end of the image.
- **Translation.** A vertical edge looks the same wherever it appears. A detector that has learned it at one position should work at every position.

A convolutional layer uses both. Each output value depends only on a small window of the input, and all windows share the same weights. The saving is large: mapping a 28×28 image to eight 28×28 feature maps takes $784 \cdot 8 \cdot 784 \approx 4.9$ million weights in a linear layer, and $8 \cdot 3 \cdot 3 = 72$ weights plus 8 biases in a 3×3 convolution.

## The convolution {#conv toc="Conv2D"}

The input of a convolutional layer has shape (batch, channels, height, width), written $(B, C, H, W)$. The layer holds $O$ kernels, each of shape $(C, k_h, k_w)$, and one bias per kernel. For every output position it places a kernel on the input, multiplies element by element and sums over all channels and kernel positions:

$$
Y_{n,o,i,j} = b_o + \sum_{c=1}^{C}\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} W_{o,c,u,v}\; X_{n,\,c,\,s i + u,\,s j + v}
$$

Here $n$ indexes the image in the batch, $b_o$ is the bias of kernel $o$ and $s$ is the **stride**, the step between neighbouring positions. Strictly speaking this is a cross-correlation, since the kernel is not flipped, but as the kernel is learned the distinction does not matter, and every deep-learning library computes it this way.

::: widget cnn-01-conv

**Padding** adds $p$ rows and columns of zeros around the input before the kernel slides over it. Without it the output shrinks with every layer and the edge pixels take part in fewer sums than the rest. The output size follows from counting how many positions the kernel fits into:

$$
H_\text{out} = \left\lfloor \frac{H + 2p - k_h}{s} \right\rfloor + 1
$$

A 3×3 kernel with $p = 1$ and $s = 1$ keeps the size; $s = 2$ halves it.

> [!abstract] Key points
> - Every output value is a dot product of the kernel with one window of the input, summed over all input channels.
> - The same weights are used at every position.
> - $H_\text{out} = \lfloor (H + 2p - k)/s \rfloor + 1$.

## im2col {#im2col}

Written as above, a convolution is six nested loops, and loops in Python are slow. nabla instead turns it into one matrix product, which NumPy hands to an optimized linear-algebra library. The trick is called **im2col** (image to columns): every window the kernel visits is copied into one column of a matrix.

::: widget cnn-02-im2col

For the general case the two matrices have the shapes

$$
X_\text{col} \in \mathbb R^{(C k_h k_w) \times (B H_\text{out} W_\text{out})}, \qquad
W_\text{flat} \in \mathbb R^{O \times (C k_h k_w)}
$$

and the whole layer is

$$
Y_\text{flat} = W_\text{flat}\, X_\text{col} + b \;\in\; \mathbb R^{O \times (B H_\text{out} W_\text{out})}
$$

In code, NumPy provides the windows as a view without copying, and one reshape lays them out as columns:

```python
windows = sliding_window_view(x_padded, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
cols = windows.transpose(1, 4, 5, 0, 2, 3).reshape(C * kh * kw, -1)

out = weight.reshape(O, -1) @ cols + bias.reshape(-1, 1)
out = out.reshape(O, B, out_h, out_w).transpose(1, 0, 2, 3)     # back to (B, O, out_h, out_w)
```

The price is memory: with stride 1 every input value appears in up to $k_h k_w$ columns, nine times for a 3×3 kernel.

## Backward through a convolution {#conv-backward toc="Backward"}

Because the forward pass is a matrix product, the backward pass starts with the two gradients of a matrix product from the [Neural networks](neural-networks.html#linear) page. With $G_\text{flat} = \partial L/\partial Y_\text{flat}$:

$$
\frac{\partial L}{\partial W_\text{flat}} = G_\text{flat}\, X_\text{col}^\top, \qquad
\frac{\partial L}{\partial X_\text{col}} = W_\text{flat}^\top\, G_\text{flat}, \qquad
\frac{\partial L}{\partial b_o} = \sum_{n,i,j} G_{n,o,i,j}
$$

The kernel gradient sums over all positions in all images, because the same weights were used at all of them. The gradient with respect to the columns still has to be turned back into the shape of the image (**col2im**): each column is added back to the window it came from.

```python
grad_cols = weight_flat.T @ grad_out_flat                   # (C*kh*kw, B*out_h*out_w)
for i in range(out_h):
    for j in range(out_w):
        grad_x_padded[:, :, i*s : i*s + kh, j*s : j*s + kw] += grad_cols[..., i, j]
```

Where windows overlap, an input value was used several times, and its gradient contributions add up, the same rule as for broadcasting on the [Autograd](../core/autograd.html#broadcasting) page. Finally the padding is cut off again. Written as a formula, the gradient with respect to the input is itself a convolution: of $G$ with the kernel rotated by 180°.

> [!abstract] Key points
> - im2col: windows become columns, the convolution becomes $W_\text{flat} X_\text{col}$.
> - Kernel gradient: $G_\text{flat} X_\text{col}^\top$, summed over all positions.
> - Input gradient: $W_\text{flat}^\top G_\text{flat}$, scattered back into the windows, overlaps add.

## Pooling {#pooling group="Downsampling and normalization"}

A pooling layer shrinks each channel by summarizing every window in a single number. Usually windows are 2×2 with stride 2, which halves height and width. Pooling has no parameters and treats every channel separately.

- **Max pooling** keeps the largest value of each window. Small shifts of a feature within its window do not change the output, which makes the network slightly less sensitive to exact positions.
- **Average pooling** keeps the mean.

The backward passes differ in kind. Average pooling is linear, so each input value receives a quarter of the gradient of its window. Max pooling is a selection: the output equals one of the inputs, so its derivative is 1 for that input and 0 for the others. The gradient is routed to the position of the maximum, which the forward pass has to remember.

::: widget cnn-03-pool

```python
# forward: remember where the maximum of each window was
self.argmax = windows.reshape(*windows.shape[:4], -1).argmax(axis=-1)

# backward: send each gradient to that position
di, dj = unravel_index(self.argmax[:, :, i, j], (k, k))
add.at(grad_x, (batch_idx, channel_idx, i*s + di, j*s + dj), grad_output[:, :, i, j])
```

`add.at` instead of plain assignment matters once windows overlap (stride smaller than the window): a value can then be the maximum of two windows and must receive both gradients.

## Batch normalization {#batchnorm toc="BatchNorm"}

During training the distribution of each layer's inputs keeps changing, because the layers before it change. Batch normalization pins down the scale of every channel: it normalizes the values of each channel across the batch and all positions, $N = B H W$ values per channel, and then lets the network rescale them with two learned parameters per channel:

$$
\mu_c = \frac1N \sum x, \qquad
\sigma_c^2 = \frac1N \sum (x - \mu_c)^2, \qquad
\hat x = \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad
y = \gamma_c\, \hat x + \beta_c
$$

The sums run over batch, height and width of channel $c$; $\epsilon$ avoids a division by zero. If normalization turns out to be harmful for a channel, the network can undo it with $\gamma_c = \sigma_c$ and $\beta_c = \mu_c$. Why exactly this helps is still debated, but in practice networks with batch normalization tolerate larger learning rates and are less sensitive to initialization.

### The backward pass

$\mu_c$ and $\sigma_c^2$ depend on every value of the channel, so each output depends on all inputs of its channel, not only on its own. The chain rule therefore has three paths to each $x$: directly through $\hat x$, through the mean and through the variance. Collected, with $g = \partial L/\partial y$ and sums over the $N$ values of the channel:

$$
\frac{\partial L}{\partial \beta_c} = \sum g, \qquad
\frac{\partial L}{\partial \gamma_c} = \sum g\, \hat x, \qquad
\frac{\partial L}{\partial x} = \frac{\gamma_c}{N\sqrt{\sigma_c^2 + \epsilon}} \Bigl( N g - \sum g - \hat x \sum g\, \hat x \Bigr)
$$

The two subtracted terms remove from the gradient whatever would shift all values of the channel together or scale them together, because the normalization cancels exactly those changes. nabla computes the same result step by step, first $\partial L/\partial \sigma^2$, then $\partial L/\partial \mu$, then $\partial L/\partial x$, following the graph of the formula; the test suite checks it against a numerical gradient.

## Training and evaluation {#train-eval toc="Train and eval"}

In training mode the statistics come from the current batch. That makes the output of one sample depend on the other samples in the batch, which is acceptable during training but not when the model is used: a prediction must not change with the company it is computed in. The layer therefore keeps **running statistics**, an exponential moving average over the training batches with momentum $m = 0.1$:

$$
\mu_\text{run} \leftarrow (1 - m)\,\mu_\text{run} + m\,\mu_c, \qquad
\sigma^2_\text{run} \leftarrow (1 - m)\,\sigma^2_\text{run} + m\,\sigma_c^2
$$

In evaluation mode (`model.eval()`) the layer normalizes with these fixed values instead.

::: widget cnn-04-batchnorm

In nabla the training mode is the `BatchNorm2D` operation with its hand-derived backward pass. The evaluation mode needs no operation of its own: with fixed statistics the layer is an ordinary affine function, written with existing tensor operations.

```python
if self.training:
    out = BatchNorm2DFunction.apply(x, self.gamma, self.beta, eps=self.eps)
    self.running_mean = (1 - m) * self.running_mean + m * x.data.mean(axis=(0, 2, 3))
    self.running_var = (1 - m) * self.running_var + m * x.data.var(axis=(0, 2, 3))
    return out
return (x - running_mean) / sqrt(running_var + eps) * gamma + beta
```

> [!abstract] Key points
> - Per channel: normalize over batch and positions, then $y = \gamma \hat x + \beta$.
> - Every output depends on the whole batch, hence the two correction terms in the gradient.
> - Training uses batch statistics, evaluation the running averages: call `model.eval()` before predicting.

## A CNN for MNIST {#mnist group="Example" toc="CNN on MNIST"}

[`examples/mnist_cnn.py`](https://github.com/maaaax1611/nabla/blob/main/examples/mnist_cnn.py) classifies handwritten digits, 28×28 grayscale images in ten classes, with two convolution blocks and one linear layer:

```python
def forward(self, x):
    x = self.conv1(x).relu().max_pool2d(kernel_size=2)   # 28×28 → 14×14
    x = self.bn1(x)
    x = self.conv2(x).relu().max_pool2d(kernel_size=2)   # 14×14 → 7×7
    x = self.bn2(x)
    x = x.reshape((x.data.shape[0], -1))                 # flatten
    return self.fc(x)
```

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| input | (B, 1, 28, 28) | |
| `Conv2D(1 → 8, 3×3, padding 1)`, ReLU | (B, 8, 28, 28) | 80 |
| `max_pool2d(2)` | (B, 8, 14, 14) | 0 |
| `BatchNorm2D(8)` | (B, 8, 14, 14) | 16 |
| `Conv2D(8 → 16, 3×3, padding 1)`, ReLU | (B, 16, 14, 14) | 1 168 |
| `max_pool2d(2)` | (B, 16, 7, 7) | 0 |
| `BatchNorm2D(16)` | (B, 16, 7, 7) | 32 |
| flatten | (B, 784) | 0 |
| `Linear(784 → 10)` | (B, 10) | 7 850 |
| total | | 9 146 |

The pattern is typical: each block halves the resolution and doubles the number of channels, trading *where* for *what*. The convolutions keep the size (3×3 with padding 1), so all downsampling happens in the pooling layers. More than 85 % of the parameters sit in the final linear layer; the convolutions get by with about 1 250 because they share their weights across all positions.

The training loop is the one from the [Neural networks](neural-networks.html#xor) page, with `CrossEntropyLoss`, `Adam` and mini-batches from a `DataLoader`. Two calls matter here: `net.train()` before each epoch and `net.eval()` before measuring accuracy, so that the batch normalization layers switch between batch and running statistics.

```bash
uv run python examples/mnist_cnn.py
```
