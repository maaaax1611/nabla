"""Downloads and parses MNIST into small numpy arrays, cached locally.

Not part of the nabla package itself - this is dataset-specific glue for the
mnist_cnn.py example, kept separate from the library like a real project
would keep its dataset loading code next to the training script that uses it.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np
from numpy.typing import NDArray

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, cache_dir: str) -> str:
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(MNIST_BASE_URL + filename, path)
    return path


def _read_idx_images(path: str) -> NDArray:
    with gzip.open(path, "rb") as f:
        _, count, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: str) -> NDArray:
    with gzip.open(path, "rb") as f:
        _, count = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist(
    n_train: int = 3000,
    n_test: int = 500,
    cache_dir: str = os.path.join(os.path.dirname(__file__), ".mnist_cache"),
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load a random subset of MNIST, downloading and caching it if needed.

    Returns:
        X_train: (n_train, 1, 28, 28) float32 in [0, 1]
        y_train: (n_train,) int64 class labels in [0, 9]
        X_test:  (n_test, 1, 28, 28) float32 in [0, 1]
        y_test:  (n_test,) int64 class labels in [0, 9]
    """
    paths = {key: _download(filename, cache_dir) for key, filename in MNIST_FILES.items()}

    train_images = _read_idx_images(paths["train_images"])
    train_labels = _read_idx_labels(paths["train_labels"])
    test_images = _read_idx_images(paths["test_images"])
    test_labels = _read_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_images), size=min(n_train, len(train_images)), replace=False)
    test_idx = rng.choice(len(test_images), size=min(n_test, len(test_images)), replace=False)

    X_train = train_images[train_idx].astype(np.float32)[:, None, :, :] / 255.0
    y_train = train_labels[train_idx].astype(np.int64)
    X_test = test_images[test_idx].astype(np.float32)[:, None, :, :] / 255.0
    y_test = test_labels[test_idx].astype(np.int64)

    return X_train, y_train, X_test, y_test
