from __future__ import annotations

import numpy as np

from nabla.optim.optimizer import Optimizer
from nabla.regularizers import Regularizer
from nabla.tensor import Tensor


class Adam(Optimizer):
    """Adam optimizer.

    Args:
        parameters: List of tensors to optimize.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        regularizers: Optional list of regularizers (e.g. L1/L2 weight decay).
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        regularizers: list[Regularizer] | None = None,
    ) -> None:
        super().__init__(parameters, lr, regularizers)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0

    def step(self) -> None:
        """Update paramters"""
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = self._regularized_grad(param)
                # Update biased first moment estimate
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
                # Update biased second raw moment estimate
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
