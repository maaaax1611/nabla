from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _fan_in_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    """Compute fan_in/fan_out for a weight shape.

    Supports Linear weights (in_features, out_features) and Conv2D
    weights (out_channels, in_channels, kh, kw).
    """
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 4:
        out_channels, in_channels, kh, kw = shape
        receptive_field = kh * kw
        fan_in = in_channels * receptive_field
        fan_out = out_channels * receptive_field
    else:
        raise ValueError(f"Cannot compute fan_in/fan_out for shape {shape}.")
    return fan_in, fan_out


def he(shape: tuple[int, ...]) -> NDArray:
    """He (Kaiming) normal initialization, suitable for ReLU activations."""
    fan_in, _ = _fan_in_fan_out(shape)
    return (np.random.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def xavier(shape: tuple[int, ...]) -> NDArray:
    """Xavier (Glorot) uniform initialization, suitable for tanh/sigmoid activations."""
    fan_in, fan_out = _fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)


def zeros(shape: tuple[int, ...]) -> NDArray:
    """Initialize all weights to zero (rarely useful, e.g. for debugging)."""
    return np.zeros(shape, dtype=np.float32)
