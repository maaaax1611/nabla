from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nabla.backend import get_array_module, get_default_dtype, to_device
from nabla.function import Function
from nabla.ops.activation import GELU, ReLU, Sigmoid, Tanh
from nabla.ops.basic import Add, Divide, Exp, Log, Multiply, Neg, Pow, Subtract
from nabla.ops.conv import Conv2D
from nabla.ops.dropout import Dropout
from nabla.ops.layernorm import LayerNorm
from nabla.ops.pooling import AvgPool2D, MaxPool2D
from nabla.ops.reduce import Mean, Sum
from nabla.ops.slice import Slice
from nabla.ops.softmax import Softmax
from nabla.ops.transform import MatMul, Reshape, Transpose


Scalar = int | float


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
        if isinstance(data, xp.ndarray):
            self.data: NDArray = data
        else:
            self.data = np.array(data)
            # Python floats / lists carry no dtype of their own, so they
            # get the library default (see `nabla.set_default_dtype`).
            # NumPy scalars (the result of e.g. `np.sum` inside an op)
            # already have one and keep it.
            if not isinstance(data, np.generic) and self.data.dtype.kind == "f":
                self.data = self.data.astype(get_default_dtype(), copy=False)
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

        # topological sort of the computation graph - iterative (explicit
        # stack), not a recursive closure: a recursive nested function
        # refers to itself through its own closure cell, which is a
        # reference cycle in its own right. That cycle would keep the
        # `backward()` call frame - and therefore the `topo` list holding
        # every tensor in the graph - alive until Python's cyclic garbage
        # collector happened to run, regardless of the _ctx/_prev
        # clearing below. Each stack entry is (tensor, expanded) - the
        # first time a tensor is popped it gets pushed back with
        # expanded=True and its parents get pushed on top of it; the
        # second time (now on top of all its parents), it's appended to
        # topo - the same post-order the recursive version produced.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            tensor, expanded = stack.pop()
            if expanded:
                topo.append(tensor)
            elif id(tensor) not in visited:
                visited.add(id(tensor))
                stack.append((tensor, True))
                for parent in tensor._prev:
                    stack.append((parent, False))

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

        # Every non-leaf tensor holds a Function (_ctx), and that Function
        # holds tensors right back via saved_tensors/_prev - a reference
        # cycle that plain refcounting can never collect on its own. Left
        # in place, each step's graph (and all the GPU memory it holds)
        # only gets freed whenever Python's cyclic GC happens to run,
        # which is not every step - on GPU that shows up as the memory
        # pool ballooning for a few steps, then a big stall while CuPy
        # allocates fresh blocks, repeating every time the GC finally
        # catches up. Breaking the cycle here (like PyTorch's default
        # retain_graph=False) frees each step's graph immediately instead.
        for tensor in topo:
            tensor._ctx = None
            tensor._prev = ()

    # Tell NumPy not to handle `ndarray <op> Tensor` itself (it would try
    # to broadcast over the Tensor as an object array) - returning
    # NotImplemented instead makes Python fall through to our __r*__ methods.
    __array_ufunc__ = None

    def _wrap(self, other: Tensor | Scalar | ArrayLike) -> Tensor:
        """Turn the other operand of a binary op into a Tensor.

        Lets `x * 2`, `1 - x` and `x + np_array` work without the caller
        wrapping constants by hand. The constant lands on this tensor's
        device and, for float tensors, takes this tensor's dtype - so a
        float32 graph isn't silently promoted to float64 by a literal.
        """
        if isinstance(other, Tensor):
            return other
        xp = get_array_module(self.data)
        dtype = self.data.dtype if self.data.dtype.kind == "f" else None
        return Tensor(xp.asarray(other, dtype=dtype))

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        return Add.apply(self, self._wrap(other))

    def __radd__(self, other: Scalar) -> Tensor:
        return Add.apply(self._wrap(other), self)

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        return Subtract.apply(self, self._wrap(other))

    def __rsub__(self, other: Scalar) -> Tensor:
        return Subtract.apply(self._wrap(other), self)

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        return Multiply.apply(self, self._wrap(other))

    def __rmul__(self, other: Scalar) -> Tensor:
        return Multiply.apply(self._wrap(other), self)

    def __truediv__(self, other: Tensor | Scalar) -> Tensor:
        return Divide.apply(self, self._wrap(other))

    def __rtruediv__(self, other: Scalar) -> Tensor:
        return Divide.apply(self._wrap(other), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __pow__(self, exponent: Scalar) -> Tensor:
        """Raise to a constant power, e.g. `x ** 2` or `x ** 0.5`."""
        if isinstance(exponent, Tensor):
            raise TypeError("Tensor exponents are not supported - use a Python number.")
        return Pow.apply(self, exponent=exponent)

    def __matmul__(self, other: Tensor) -> Tensor:
        return MatMul.apply(self, other)

    def __getitem__(self, key) -> Tensor:
        """Basic indexing (ints, slices, Ellipsis, None), like numpy - e.g.
        `x[:, 0]`, `x[1:3]`, `x[..., :2]`. No boolean/fancy-array indices.
        """
        return Slice.apply(self, key=key)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum over `axis` (all elements, giving a scalar, if axis is None)."""
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Average over `axis` (all elements, giving a scalar, if axis is None)."""
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def exp(self) -> Tensor:
        """Element-wise exponential."""
        return Exp.apply(self)

    def log(self) -> Tensor:
        """Element-wise natural logarithm."""
        return Log.apply(self)

    def sqrt(self) -> Tensor:
        """Element-wise square root."""
        return Pow.apply(self, exponent=0.5)

    def astype(self, dtype) -> Tensor:
        """Convert the underlying data (and gradient, if any) to `dtype`.

        Mutates and returns self, like `.to()` - meant for leaf tensors
        (inputs, parameters), not for casting in the middle of a graph.
        """
        self.data = self.data.astype(dtype, copy=False)
        if self.grad is not None:
            self.grad = self.grad.astype(dtype, copy=False)
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

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

    def tanh(self) -> Tensor:
        """Apply tanh activation element-wise."""
        return Tanh.apply(self)

    def gelu(self) -> Tensor:
        """Apply GELU activation element-wise (tanh approximation)."""
        return GELU.apply(self)

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