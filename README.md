# nabla

[![tests](https://github.com/maaaax1611/nabla/actions/workflows/tests.yml/badge.svg)](https://github.com/maaaax1611/nabla/actions/workflows/tests.yml)
[![docs](https://github.com/maaaax1611/nabla/actions/workflows/docs.yml/badge.svg)](https://maaaax1611.github.io/nabla/)

**Documentation:** https://maaaax1611.github.io/nabla/

A small automatic differentiation library built on NumPy, modeled on
PyTorch's design: a define-by-run computational graph, reverse-mode
autodiff, and `nn.Module` / `Optimizer` layers on top — enough to train a
CNN, a character-level Transformer and a Vision Transformer. Optional GPU
support via CuPy.

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

## What's in it

| Area | Pieces |
| --- | --- |
| Core | `Tensor`, `Function`, `backward()`, broadcasting, slicing, `no_grad()`, configurable default dtype |
| Ops | `+ - * / **`, `matmul`, `exp`, `log`, `sqrt`, `sum` / `mean` over an axis, `reshape`, `transpose`, `concat` |
| Layers | `Linear`, `Conv2D`, max / average pooling, `BatchNorm2D`, `LayerNorm`, `Dropout`, `Embedding`, `PositionalEncoding`, `Attention`, `TransformerBlock`, `VisionTransformer` |
| Activations | ReLU, Sigmoid, Tanh, GELU, Softmax |
| Losses | `MSELoss`, `CrossEntropyLoss` |
| Training | `SGD`, `Adam`, L1 / L2 regularizers, LR schedules, gradient clipping, checkpointing, metric logging, `DataLoader` |
| Backend | NumPy on CPU, CuPy on GPU (`uv sync --extra gpu`, then `.to("cuda")`) |

## Run it

```bash
uv run pytest                                      # test suite
uv run python examples/xor.py                      # tiny MLP on XOR
uv run python examples/mnist_cnn.py                # small CNN on a MNIST subset
uv run python examples/mnist_vit.py                # Vision Transformer on MNIST
uv run python examples/shakespeare_transformer.py  # character-level Transformer
```

## How it works

The [documentation](https://maaaax1611.github.io/nabla/) explains each building block on its own page:
the math, the code that implements it and figures you can step through.

- [Autograd](https://maaaax1611.github.io/nabla/core/autograd.html): the computational graph, the
  backward pass, broadcasting, gradient checks
- [Neural networks](https://maaaax1611.github.io/nabla/networks/neural-networks.html): modules, linear
  layers, activations, initialization, softmax cross-entropy, XOR
- [Convolutional networks](https://maaaax1611.github.io/nabla/networks/cnn.html): convolution, im2col,
  pooling, batch normalization, a CNN for MNIST
- [Transformers](https://maaaax1611.github.io/nabla/networks/transformer.html): embeddings, positional
  encoding, attention, LayerNorm, the Transformer block, Shakespeare, ViT
- [Training](https://maaaax1611.github.io/nabla/training/training.html): mini-batches, SGD, Adam, weight
  decay, dropout, learning-rate schedules, gradient clipping, checkpoints, GPU

The sources are in [`docs/`](docs/), built with
[doc.md](https://github.com/maaaax1611/doc.md); every push to `main` that
changes them republishes the site. To work on them locally:

```bash
uv run --group docs docmd serve docs     # http://127.0.0.1:8000
```

Bugs that came up along the way and how they were fixed are collected in the
[engineering report](reports/engineering-report.md).
