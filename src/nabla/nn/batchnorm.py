from __future__ import annotations

import numpy as np

from nabla.nn.module import Module
from nabla.ops.batchnorm import BatchNorm2D as BatchNorm2DFunction
from nabla.tensor import Tensor


class BatchNorm2D(Module):
    """2D batch normalization layer.

    In training mode, normalizes using the current batch's statistics (see
    ``ops.batchnorm.BatchNorm2D``) and updates running mean/variance via an
    exponential moving average. In eval mode, normalizes using those running
    statistics instead, so a single input can be normalized independently
    of any batch it's part of.

    Args:
        num_features: Number of channels C in the input (batch, C, H, W).
        eps: Small constant added to the variance for numerical stability.
        momentum: Weight given to the current batch when updating the
            running statistics: running = (1 - momentum) * running + momentum * batch.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)

        # running statistics: plain arrays, not learnable parameters
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")
        if x.data.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} channels, but got {x.data.shape[1]}."
            )

        if self.training:
            out = BatchNorm2DFunction.apply(x, self.gamma, self.beta, eps=self.eps)
            # ctx (the Function instance) is reachable via out._ctx and still
            # holds the batch statistics computed during forward
            batch_mean = out._ctx.batch_mean.reshape(-1)
            batch_var = out._ctx.batch_var.reshape(-1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            return out

        # eval mode: normalize with the running statistics via plain Tensor
        # ops, so this stays differentiable w.r.t. gamma/beta without needing
        # a separate autograd Function
        mean = Tensor(self.running_mean.reshape(1, -1, 1, 1))
        std = Tensor(np.sqrt(self.running_var + self.eps).reshape(1, -1, 1, 1))
        gamma = self.gamma.reshape((1, self.num_features, 1, 1))
        beta = self.beta.reshape((1, self.num_features, 1, 1))
        return (x - mean) / std * gamma + beta
