"""Free-function counterparts to the Tensor methods of the same name.

For ops where one operand naturally reads as "self" (`x.relu()`,
`x.softmax()`), the Tensor method is usually the more natural call. This
module exists for the other case: ops where the arguments are symmetric or
there's no obviously-primary Tensor (`matmul(Q, K)`), where hanging the
call off one arbitrary argument as `self` feels arbitrary rather than
natural - the same reasoning PyTorch's `torch.nn.functional` module (and
free functions like `torch.matmul`) is built on. Every function here is a
thin wrapper around the same `Function.apply()` the Tensor method calls -
nothing here does anything the Tensor methods couldn't already do.
"""

from __future__ import annotations

import numpy as np

from nabla.ops.activation import ReLU, Sigmoid
from nabla.ops.conv import Conv2D
from nabla.ops.dropout import Dropout
from nabla.ops.layernorm import LayerNorm
from nabla.ops.pooling import AvgPool2D, MaxPool2D
from nabla.ops.reduce import Mean, Sum
from nabla.ops.softmax import Softmax
from nabla.ops.transform import MatMul, Reshape, Transpose
from nabla.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation element-wise."""
    return ReLU.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation element-wise."""
    return Sigmoid.apply(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Apply softmax over the given axis (default: the last axis)."""
    return Softmax.apply(x, axis=axis)


def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    """Apply inverted dropout element-wise, zeroing each element with probability p."""
    return Dropout.apply(x, p=p)


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize over the last axis, using per-sample statistics (no batch dependency)."""
    return LayerNorm.apply(x, gamma, beta, eps=eps)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication (batched over any leading axes)."""
    return MatMul.apply(a, b)


def reshape(x: Tensor, new_shape: tuple[int, ...]) -> Tensor:
    """Reshape a tensor to a new shape (total size must stay the same)."""
    return Reshape.apply(x, new_shape=new_shape)


def transpose(x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
    """Permute a tensor's axes. If axes is None, reverses all axes."""
    return Transpose.apply(x, axes=axes)


def conv2d(x: Tensor, weight: Tensor, bias: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
    """2D convolution.

    Args:
        weight: Kernel of shape (out_channels, in_channels, kh, kw).
        bias: Per-output-channel bias of shape (out_channels,).
        stride: Step size of the sliding window.
        padding: Zero-padding added to both sides of the H/W axes.
    """
    return Conv2D.apply(x, weight, bias, stride=stride, padding=padding)


def max_pool2d(x: Tensor, kernel_size: int, stride: int | None = None) -> Tensor:
    """2D max pooling. stride defaults to kernel_size (non-overlapping windows)."""
    return MaxPool2D.apply(x, kernel_size=kernel_size, stride=stride)


def avg_pool2d(x: Tensor, kernel_size: int, stride: int | None = None) -> Tensor:
    """2D average pooling. stride defaults to kernel_size (non-overlapping windows)."""
    return AvgPool2D.apply(x, kernel_size=kernel_size, stride=stride)


def sum(x: Tensor) -> Tensor:
    """Reduce all elements to a scalar by summation."""
    return Sum.apply(x)


def mean(x: Tensor) -> Tensor:
    """Reduce all elements to a scalar by averaging."""
    return Mean.apply(x)


def _swap_last_two_axes(t: Tensor) -> Tensor:
    """Transpose only the trailing two axes, leaving any leading (batch,
    heads, ...) axes untouched - what K^T means for batched attention."""
    ndim = t.data.ndim
    axes = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return transpose(t, axes)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: np.ndarray | None = None) -> Tensor:
    """out = softmax(Q @ K^T / sqrt(d_k)) @ V.

    Shapes:
        Q: (*, seq_len_q, d_k)  - any number of leading axes (batch, later
           also num_heads), then query positions, then feature dim
        K: (*, seq_len_k, d_k)  - same leading axes as Q
        V: (*, seq_len_k, d_v)  - same key/value positions as K
        out: (*, seq_len_q, d_v)
        mask: optional, broadcastable against (*, seq_len_q, seq_len_k) -
              added *before* the softmax (e.g. -1e9 at positions that must
              not be attended to). Plain ndarray, not a Tensor - nothing
              meaningful to differentiate w.r.t. a fixed mask.

    Composed entirely from other already-differentiable ops in this module
    (matmul, transpose, softmax, elementwise multiply/add) rather than its
    own Function with a hand-derived backward - autodiff figures out the
    backward pass automatically by walking the graph these calls build.
    """
    d_k = Q.data.shape[-1]

    scores = matmul(Q, _swap_last_two_axes(K))
    scores = scores * Tensor(np.array(1 / np.sqrt(d_k)))
    if mask is not None:
        scores = scores + Tensor(mask)

    weights = softmax(scores, axis=-1)
    return matmul(weights, V)
