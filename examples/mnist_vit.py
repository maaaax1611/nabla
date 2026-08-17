"""MNIST digit classification with a Vision Transformer.

See docs/17-vision-transformer.md for the architecture: images become a
sequence of 4x4 patches, a learnable CLS token is prepended, a learned
positional embedding is added, and the same TransformerBlock stack used
for text (docs/13-transformer-block.md) runs over the result - no mask,
since a full image has no "future" to hide.

Runs on plain NumPy (no GPU), so both the model and training budget here
are kept small enough to finish in a few minutes on CPU. This is a small
proof-of-concept: ViT is known to need much more data/compute than a CNN
to reach comparable accuracy from scratch (it lacks a CNN's built-in
translation-equivariance bias), so don't expect this to beat
examples/mnist_cnn.py's accuracy - the point is exercising every ViT
building block end to end, not chasing state of the art.
"""

import numpy as np
from mnist_data import load_mnist

from nabla.data.dataloader import DataLoader
from nabla.nn.loss import CrossEntropyLoss
from nabla.nn.module import Module
from nabla.nn.vit import VisionTransformer
from nabla.optim.adam import Adam
from nabla.tensor import Tensor


def accuracy(net: Module, X: Tensor, y: np.ndarray) -> float:
    logits = net(X)
    predicted = np.argmax(logits.data, axis=1)
    return float(np.mean(predicted == y))


def main() -> None:
    print("Loading MNIST (downloads + caches on first run)...")
    X_train, y_train, X_test, y_test = load_mnist(n_train=3000, n_test=500)
    train_loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
    X_test_t = Tensor(X_test)

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    model = VisionTransformer(
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    print(f"Model parameters: {sum(p.data.size for p in model.parameters()):,}")

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data) * len(X_batch.data)

        epoch_loss /= len(X_train)

        model.eval()
        test_acc = accuracy(model, X_test_t, y_test)
        print(f"Epoch {epoch + 1:2d} | Train loss: {epoch_loss:.4f} | Test accuracy: {test_acc:.2%}")

    model.eval()
    print("\nSample predictions:")
    logits = model(X_test_t)
    predicted = np.argmax(logits.data, axis=1)
    for i in range(10):
        print(f"  true={y_test[i]}  predicted={predicted[i]}")


if __name__ == "__main__":
    main()
