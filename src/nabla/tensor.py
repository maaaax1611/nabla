from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nabla.function import Function
from nabla.ops.activation import ReLU, Sigmoid
from nabla.ops.basic import Add, Divide, Multiply, Subtract
from nabla.ops.reduce import Mean, Sum
from nabla.ops.transform import MatMul, Reshape, Transpose


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    Args:
        data: The underlying data, will be stored as-is (should be a numpy array).
        requires_grad: If True, gradients will be computed for this tensor.

    Attributes:
        data: The numpy array holding the values.
        grad: The accumulated gradient, same shape as data.
    """

    def __init__(self, data: ArrayLike, requires_grad: bool = False) -> None:
        self.data: NDArray = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad: NDArray | None = None
        self._ctx: Function | None = None
        self._prev: tuple[Tensor, ...] = ()

    def backward(self, grad: NDArray | None = None) -> None:
        """Compute gradients via reverse-mode automatic differentiation.

        Args:
            grad: The initial gradient. If None, defaults to ones with the same shape as data.
        """
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = grad

        # topological sort of the computation graph
        topo: list[Tensor] = []
        visited: set[int] = set()

        def topo_sort(tensor: Tensor) -> None:
            if id(tensor) not in visited:
                visited.add(id(tensor))
                for parent in tensor._prev:
                    topo_sort(parent)
                topo.append(tensor)

        topo_sort(self)

        # propagate gradients backward through the sorted graph
        for tensor in reversed(topo):
            if tensor._ctx:
                grads = tensor._ctx.backward(tensor.grad)
                for parent, g in zip(tensor._prev, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = g
                        else:
                            parent.grad = parent.grad + g

    def __add__(self, other: Tensor) -> Tensor:
        return Add.apply(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        return Subtract.apply(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return Multiply.apply(self, other)

    def __truediv__(self, other: Tensor) -> Tensor:
        return Divide.apply(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return MatMul.apply(self, other)

    def sum(self) -> Tensor:
        """Reduce all elements to a scalar by summation."""
        return Sum.apply(self)

    def mean(self) -> Tensor:
        """Reduce all elements to a scalar by averaging."""
        return Mean.apply(self)

    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication with another tensor."""
        return MatMul.apply(self, other)

    def reshape(self, new_shape: tuple[int, ...]) -> Tensor:
        """Reshape the tensor to a new shape (total size must stay the same)."""
        return Reshape.apply(self, new_shape=new_shape)

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        """Permute the tensor's axes. If axes is None, reverses all axes."""
        return Transpose.apply(self, axes=axes)

    @property
    def T(self) -> Tensor:
        """Reverse all axes, mirroring numpy's ``.T`` behavior."""
        return Transpose.apply(self)

    def relu(self) -> Tensor:
        """Apply ReLU activation element-wise."""
        return ReLU.apply(self)

    def sigmoid(self) -> Tensor:
        """Apply sigmoid activation element-wise."""
        return Sigmoid.apply(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"