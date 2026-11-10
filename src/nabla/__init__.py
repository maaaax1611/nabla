"""nabla - a small reverse-mode automatic differentiation library on NumPy.

    import nabla
    from nabla import nn

    x = nabla.Tensor([[1.0, 2.0]], requires_grad=True)
    loss = ((x * 2) ** 2).sum()
    loss.backward()
"""

from nabla.backend import get_default_dtype, gpu_available, set_default_dtype
from nabla.function import Function, is_grad_enabled, no_grad
from nabla.tensor import Tensor
from nabla import functional, nn, optim

__all__ = [
    "Function",
    "Tensor",
    "functional",
    "get_default_dtype",
    "gpu_available",
    "is_grad_enabled",
    "nn",
    "no_grad",
    "optim",
    "set_default_dtype",
]
