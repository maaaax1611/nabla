import numpy as np
from mnist_data import load_mnist

from nabla.data.dataloader import DataLoader
from nabla.nn.batchnorm import BatchNorm2D
from nabla.nn.conv import Conv2D
from nabla.nn.linear import Linear
from nabla.nn.loss import CrossEntropyLoss
from nabla.nn.module import Module
from nabla.optim.adam import Adam
from nabla.tensor import Tensor


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(num_features=8)
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(num_features=16)
        self.fc = Linear(in_features=16 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = x.max_pool2d(kernel_size=2)  # 28x28 -> 14x14
        x = self.bn1(x)

        x = self.conv2(x).relu()
        x = x.max_pool2d(kernel_size=2)  # 14x14 -> 7x7
        x = self.bn2(x)

        x = x.reshape((x.data.shape[0], -1))  # flatten to (batch, 16*7*7)
        return self.fc(x)


def accuracy(net: Module, X: Tensor, y: np.ndarray) -> float:
    logits = net(X)
    predicted = np.argmax(logits.data, axis=1)
    return float(np.mean(predicted == y))


# data
print("Loading MNIST (downloads + caches on first run)...")
X_train, y_train, X_test, y_test = load_mnist(n_train=3000, n_test=500)
train_loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
X_test_t = Tensor(X_test)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# setup
net = SimpleCNN()
criterion = CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=1e-3)

# training loop
epochs = 5
for epoch in range(epochs):
    net.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        logits = net(X_batch)
        loss = criterion(logits, y_batch)

        net.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.data) * len(X_batch.data)

    epoch_loss /= len(X_train)

    net.eval()
    test_acc = accuracy(net, X_test_t, y_test)
    print(f"Epoch {epoch + 1:2d} | Train loss: {epoch_loss:.4f} | Test accuracy: {test_acc:.2%}")

# results
net.eval()
print("\nSample predictions:")
logits = net(X_test_t)
predicted = np.argmax(logits.data, axis=1)
for i in range(10):
    print(f"  true={y_test[i]}  predicted={predicted[i]}")
