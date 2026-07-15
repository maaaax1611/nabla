import numpy as np
from nabla.tensor import Tensor
from nabla.nn.module import Module
from nabla.nn.linear import Linear
from nabla.nn.loss import MSELoss
from nabla.optim.sgd import SGD


class XORNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 8)
        self.fc2 = Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.sigmoid()
        return x


# data
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64))
y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float64))

# setup
net = XORNet()
criterion = MSELoss()
optimizer = SGD(net.parameters(), lr=0.1)

# training loop
epochs = 1000
for epoch in range(epochs):
    pred = net(X)
    loss = criterion(pred, y)

    net.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")

# results
print("\nPredictions:")
pred = net(X)
for i in range(4):
    print(f"  {X.data[i]} -> {pred.data[i][0]:.4f} (expected: {y.data[i][0]})")
