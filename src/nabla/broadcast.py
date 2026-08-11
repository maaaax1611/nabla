import numpy as np
from numpy.typing import NDArray


def unbroadcast(grad: NDArray, target_shape: tuple[int, ...]) -> NDArray:
    """Reduce a gradient back to its original shape after broadcasting.

    Args:
        grad: The gradient array that may have been broadcast to a larger shape.
        target_shape: The original shape to reduce back to.

    Returns:
        The gradient reduced to target_shape by summing over broadcast dimensions.
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad