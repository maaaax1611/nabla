from nabla.nn.module import Module
from nabla.tensor import Tensor
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # for now we use He-init but we can modularize this later (add kaiming, xavier, etc)
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with last dimension {self.in_features}, but got {x.data.shape[-1]}.")
        
        return x.matmul(self.weight) + self.bias