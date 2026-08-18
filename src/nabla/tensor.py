from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nabla.backend import get_array_module, to_device
from nabla.function import Function
from nabla.ops.activation import ReLU, Sigmoid
from nabla.ops.basic import Add, Divide, Multiply, Subtract
from nabla.ops.conv import Conv2D
from nabla.ops.dropout import Dropout
from nabla.ops.layernorm import LayerNorm
from nabla.ops.pooling import AvgPool2D, MaxPool2D
from nabla.ops.reduce import Mean, Sum
from nabla.ops.slice import Slice
from nabla.ops.softmax import Softmax
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
        xp = get_array_module(data)
        self.data: NDArray = data if isinstance(data, xp.ndarray) else np.array(data)
        self.requires_grad = requires_grad
        self.grad: NDArray | None = None
        self._ctx: Function | None = None
        self._prev: tuple[Tensor, ...] = ()

    def to(self, device: str) -> Tensor:
        """Move this tensor's data (and gradient, if any) to "cpu" or "cuda".

        Mutates and returns self, mirroring PyTorch's in-place-by-default
        `.to()` for leaf tensors - there's no autograd graph to preserve
        across a device move here, so there's nothing an out-of-place
        version would need to protect.
        """
        self.data = to_device(self.data, device)
        if self.grad is not None:
            self.grad = to_device(self.grad, device)
        return self

    def backward(self, grad: NDArray | None = None) -> None:
        """Compute gradients via reverse-mode automatic differentiation.

        Args:
            grad: The initial gradient. If None, defaults to ones with the same shape as data.
        """
        if grad is None:
            xp = get_array_module(self.data)
            self.grad = xp.ones_like(self.data)
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

    def __getitem__(self, key) -> Tensor:
        """Basic indexing (ints, slices, Ellipsis, None), like numpy - e.g.
        `x[:, 0]`, `x[1:3]`, `x[..., :2]`. No boolean/fancy-array indices.
        """
        return Slice.apply(self, key=key)

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

    def conv2d(self, weight: Tensor, bias: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
        """2D convolution with this tensor as input.

        Args:
            weight: Kernel of shape (out_channels, in_channels, kh, kw).
            bias: Per-output-channel bias of shape (out_channels,).
            stride: Step size of the sliding window.
            padding: Zero-padding added to both sides of the H/W axes.
        """
        return Conv2D.apply(self, weight, bias, stride=stride, padding=padding)

    def max_pool2d(self, kernel_size: int, stride: int | None = None) -> Tensor:
        """2D max pooling. stride defaults to kernel_size (non-overlapping windows)."""
        return MaxPool2D.apply(self, kernel_size=kernel_size, stride=stride)

    def avg_pool2d(self, kernel_size: int, stride: int | None = None) -> Tensor:
        """2D average pooling. stride defaults to kernel_size (non-overlapping windows)."""
        return AvgPool2D.apply(self, kernel_size=kernel_size, stride=stride)

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

    def dropout(self, p: float = 0.5) -> Tensor:
        """Apply inverted dropout element-wise, zeroing each element with probability p."""
        return Dropout.apply(self, p=p)

    def layer_norm(self, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
        """Normalize over the last axis, using per-sample statistics (no batch dependency)."""
        return LayerNorm.apply(self, gamma, beta, eps=eps)

    def softmax(self, axis: int = -1) -> Tensor:
        """Apply softmax over the given axis (default: the last axis)."""
        return Softmax.apply(self, axis=axis)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"