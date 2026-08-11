from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nabla.tensor import Tensor


class DataLoader:
    """Iterates over a dataset in (optionally shuffled) mini-batches.

    Args:
        X: Input samples, array of shape (N, ...).
        y: Targets, array of shape (N, ...).
        batch_size: Number of samples per batch. The last batch may be smaller.
        shuffle: Whether to shuffle the sample order at the start of each iteration.
    """

    def __init__(self, X: NDArray, y: NDArray, batch_size: int = 32, shuffle: bool = True) -> None:
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Number of batches per full iteration."""
        return -(-len(self.X) // self.batch_size)  # ceil division

    def __iter__(self):
        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield Tensor(self.X[batch_idx]), Tensor(self.y[batch_idx])
