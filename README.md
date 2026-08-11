# nabla

A small automatic differentiation library built on NumPy, modeled on
PyTorch's design — define-by-run computational graph, reverse-mode autodiff,
and a growing set of ops (elementwise, matmul, Conv2D, pooling, BatchNorm,
softmax cross-entropy) with `nn.Module`/`Optimizer` layers on top.

```bash
uv run pytest              # run the test suite
uv run python examples/xor.py         # tiny MLP on XOR
uv run python examples/mnist_cnn.py   # small CNN on a MNIST subset
```

## Docs

[`docs/`](docs/README.md) walks through the math behind each op, the
derivation of its backward pass, and how that maps onto the code — start
there if you want to understand *why* something is implemented the way it
is, not just what it does.
