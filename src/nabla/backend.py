"""Array-backend dispatch: lets every op run on either NumPy (CPU) or
CuPy (GPU) without branching on device anywhere in the ops themselves.

CuPy mirrors NumPy's API almost exactly (cupy.ndarray supports the same
methods/functions as numpy.ndarray for everything this library uses), so
the standard trick - used by libraries like Chainer, which is what CuPy
was originally built for - is to look up which module a given array
already belongs to and call functions on *that* module instead of
hardcoding `np`:

    xp = get_array_module(x.data)
    return xp.exp(x.data)

`xp.exp` is `numpy.exp` for a numpy.ndarray and `cupy.exp` for a
cupy.ndarray - the op's code doesn't need to know or care which.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def gpu_available() -> bool:
    """True if CuPy is installed *and* a working CUDA device is reachable.

    Distinct from CUPY_AVAILABLE (which only checks the import): CuPy can
    be installed but still fail to find a GPU/driver at runtime, so tests
    that need an actual device should gate on this instead.
    """
    if not CUPY_AVAILABLE:
        return False
    try:
        return bool(cp.cuda.is_available())
    except Exception:
        return False


def get_array_module(*arrays: Any):
    """Return the numpy or cupy module that should be used for these arrays.

    Mirrors ``cupy.get_array_module``, but works even when CuPy isn't
    installed at all (falls back to numpy unconditionally) - so ops can
    call this without needing to guard every call site on CUPY_AVAILABLE.
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(*arrays)
    return np


def is_gpu_array(array: Any) -> bool:
    """True if `array` is a cupy.ndarray (lives on the GPU)."""
    return CUPY_AVAILABLE and isinstance(array, cp.ndarray)


def to_device(array: Any, device: str) -> Any:
    """Move a numpy/cupy array to the given device ("cpu" or "cuda").

    No-op if the array is already on the requested device.
    """
    if device == "cpu":
        return cp.asnumpy(array) if is_gpu_array(array) else np.asarray(array)
    if device == "cuda":
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not installed - install nabla with the 'gpu' extra to use device='cuda'.")
        return array if is_gpu_array(array) else cp.asarray(array)
    raise ValueError(f"Unknown device {device!r}, expected 'cpu' or 'cuda'.")
