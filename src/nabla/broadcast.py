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
    return grad