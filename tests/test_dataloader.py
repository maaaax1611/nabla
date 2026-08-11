from nabla.data.dataloader import DataLoader
from nabla.tensor import Tensor
import numpy as np


class TestDataLoader:
    def test_length_is_ceil_division(self):
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        assert len(DataLoader(X, y, batch_size=3)) == 4
        assert len(DataLoader(X, y, batch_size=5)) == 2
        assert len(DataLoader(X, y, batch_size=10)) == 1

    def test_iterating_yields_tensors(self):
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        loader = DataLoader(X, y, batch_size=4, shuffle=False)
        for batch_x, batch_y in loader:
            assert isinstance(batch_x, Tensor)
            assert isinstance(batch_y, Tensor)

    def test_last_batch_may_be_smaller(self):
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        loader = DataLoader(X, y, batch_size=4, shuffle=False)
        batch_sizes = [len(batch_x.data) for batch_x, _ in loader]
        assert batch_sizes == [4, 4, 2]

    def test_covers_every_sample_exactly_once_without_shuffle(self):
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        loader = DataLoader(X, y, batch_size=3, shuffle=False)

        seen = np.concatenate([batch_x.data.flatten() for batch_x, _ in loader])
        assert np.array_equal(seen, X.flatten())

    def test_covers_every_sample_exactly_once_with_shuffle(self):
        np.random.seed(0)
        X = np.arange(20).reshape(20, 1)
        y = np.arange(20)
        loader = DataLoader(X, y, batch_size=6, shuffle=True)

        seen = np.concatenate([batch_x.data.flatten() for batch_x, _ in loader])
        assert np.array_equal(np.sort(seen), X.flatten())

    def test_x_and_y_stay_aligned(self):
        np.random.seed(1)
        X = np.arange(20).reshape(20, 1)
        y = X.flatten() * 10  # y[i] should always equal X[i] * 10
        loader = DataLoader(X, y, batch_size=7, shuffle=True)

        for batch_x, batch_y in loader:
            assert np.array_equal(batch_x.data.flatten() * 10, batch_y.data)

    def test_shuffle_changes_batch_order_across_iterations(self):
        np.random.seed(2)
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = DataLoader(X, y, batch_size=10, shuffle=True)

        first_pass = np.concatenate([b.data.flatten() for b, _ in loader])
        second_pass = np.concatenate([b.data.flatten() for b, _ in loader])
        assert not np.array_equal(first_pass, second_pass)

    def test_mismatched_lengths_raise(self):
        X = np.arange(10).reshape(10, 1)
        y = np.arange(5)
        try:
            DataLoader(X, y, batch_size=2)
            assert False, "expected ValueError"
        except ValueError:
            pass
