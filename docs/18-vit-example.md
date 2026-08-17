# Example: MNIST Digit Classification with a Vision Transformer

## What this ties together

[`examples/mnist_vit.py`](../examples/mnist_vit.py) trains
[`VisionTransformer`](17-vision-transformer.md) on MNIST digit
classification, reusing the same [`DataLoader`](../src/nabla/data/dataloader.py)
and MNIST loading code as
[`examples/mnist_cnn.py`](../examples/mnist_cnn.py) — only the model and
training loop change.

```python
model = VisionTransformer(
    img_size=28, patch_size=4, in_channels=1, num_classes=10,
    embed_dim=64, num_heads=4, hidden_dim=128, num_layers=4, dropout=0.1,
)
```

`28 / 4 = 7`, so each 28x28 digit becomes a 7x7 grid — 49 patches, plus
the CLS token, for a sequence length of 50 fed through 4 stacked
`TransformerBlock`s.

## Training loop

Identical in shape to `mnist_cnn.py`'s: `CrossEntropyLoss`, `Adam`,
`zero_grad()` / `backward()` / `step()` per batch, `train()`/`eval()`
toggled around each epoch's evaluation pass. Nothing ViT-specific here —
once the model itself is built, "train an image classifier" looks the
same regardless of what's inside the model.

## Results

10 epochs over a 3000-image training subset (same budget as
`mnist_cnn.py`), on CPU:

```
Epoch  1 | Train loss: 1.7570 | Test accuracy: 58.20%
Epoch  5 | Train loss: 0.7496 | Test accuracy: 76.80%
Epoch 10 | Train loss: 0.5425 | Test accuracy: 81.60%
```

~2.5-3 minutes total. Don't expect this to beat `mnist_cnn.py`'s
accuracy at a comparable budget — a CNN has translation-equivariance
built into its architecture (a learned filter that recognizes an edge
works the same wherever the edge appears), which a Transformer has to
learn from data instead. This gap is a well-known, real property of
ViT, not a nabla-specific issue: the original ViT paper needed
much larger datasets (or pretraining) to match or beat CNNs. The point
of this example is exercising every ViT building block end to end —
`Conv2D`-as-patch-projection, `Concat`-prepended CLS token, learned
positional embeddings, a full Transformer stack, `MatMul`-based
token selection — not chasing state-of-the-art accuracy.

## Running it

```bash
uv run python examples/mnist_vit.py
```

MNIST downloads once and is cached under `examples/.mnist_cache/`
(git-ignored, shared with `mnist_cnn.py`) on first run.
