---
title: nabla
description: Building an automatic differentiation library from scratch.
---

# nabla

**nabla** is a small automatic differentiation library built on NumPy and
modelled on PyTorch's design: a define-by-run computational graph,
reverse-mode autodiff, and enough layers on top to train a CNN, a
character-level Transformer and a Vision Transformer.

This site is the long-form explanation of how it works. Each post takes one
topic, derives the math, and maps it onto the code that implements it —
including the bugs that showed up along the way, which are often more
instructive than the version that worked.

## The series

Read in order; each post builds on the ones before it.

| | Post | What it covers |
| --- | --- | --- |
| 1 | [Autograd from scratch](posts/01-autograd.md) | `Tensor`, `Function`, the graph, the chain rule as a topological sort, broadcasting, `no_grad`, and a memory leak hiding in a reference cycle |
| 2 | [From a neuron to an MLP](posts/02-mlp.md) | `Module`, linear layers, activations, initialization, softmax, and why softmax + cross-entropy is one op |
| 3 | [Training](posts/03-training.md) | SGD, Adam, weight decay, dropout, learning-rate schedules, gradient clipping |
| 4 | [Convolutional networks](posts/04-cnn.md) | Conv2D via im2col, pooling, BatchNorm with the full backward derivation, a CNN on MNIST |
| 5 | [Attention and the Transformer](posts/05-transformer.md) | Embeddings, positional encoding, multi-head attention, LayerNorm, the Transformer block, a Shakespeare language model |
| 6 | [Vision Transformer](posts/06-vit.md) | Patches as tokens, `Concat`, slicing, ViT vs. CNN on MNIST |
| 7 | [Engineering](posts/07-engineering.md) | GPU support via CuPy, `float32` vs. `float64`, checkpointing, logging |

## Thirty seconds of nabla

```python
import numpy as np
import nabla
from nabla import nn, optim

model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

x = nabla.Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))
y = nabla.Tensor(np.array([[1.0], [0.0]]))

for _ in range(100):
    loss = ((model(x) - y) ** 2).mean()
    model.zero_grad()
    loss.backward()
    optimizer.step()
```

The code lives at
[github.com/maaaax1611/nabla](https://github.com/maaaax1611/nabla).
