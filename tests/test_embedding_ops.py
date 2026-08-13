import numpy as np

from nabla.ops.embedding import Embedding
from nabla.tensor import Tensor


class TestEmbedding:
    def test_forward_shape(self):
        table = Tensor(np.random.randn(10, 4))
        indices = Tensor(np.array([1, 3, 5]))
        out = Embedding.apply(table, indices)
        assert out.data.shape == (3, 4)

    def test_forward_shape_with_leading_axes(self):
        table = Tensor(np.random.randn(10, 4))
        indices = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # (batch, seq_len)
        out = Embedding.apply(table, indices)
        assert out.data.shape == (2, 3, 4)

    def test_forward_matches_plain_gather(self):
        np.random.seed(0)
        table = Tensor(np.random.randn(10, 4))
        indices = Tensor(np.array([0, 9, 3, 3]))
        out = Embedding.apply(table, indices)
        assert np.array_equal(out.data, table.data[indices.data])

    def test_backward_accumulates_gradient_for_repeated_indices(self):
        table = Tensor(np.random.randn(5, 3), requires_grad=True)
        indices = Tensor(np.array([1, 1, 1]))  # same index three times
        out = Embedding.apply(table, indices)

        grad_output = np.ones_like(out.data)
        out.backward(grad_output)

        expected = np.zeros_like(table.data)
        expected[1] = 3.0  # three separate contributions, all to row 1
        assert np.array_equal(table.grad, expected)

    def test_backward_only_touches_used_rows(self):
        table = Tensor(np.random.randn(5, 3), requires_grad=True)
        indices = Tensor(np.array([0, 2]))
        out = Embedding.apply(table, indices)
        out.backward(np.ones_like(out.data))

        assert np.array_equal(table.grad[1], np.zeros(3))
        assert np.array_equal(table.grad[3], np.zeros(3))
        assert np.array_equal(table.grad[4], np.zeros(3))

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(1)
        table = Tensor(np.random.randn(6, 4), requires_grad=True)
        indices = Tensor(np.array([[0, 2, 2, 5], [1, 1, 3, 0]]))  # repeats on purpose

        def forward():
            return Embedding().forward(table, indices)

        grad_output = np.random.randn(*forward().shape)
        out = Embedding.apply(table, indices)
        out.backward(grad_output)

        eps = 1e-5
        num_grad = np.zeros_like(table.data)
        it = np.nditer(table.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = table.data[idx]

            table.data[idx] = original + eps
            out_plus = forward().copy()

            table.data[idx] = original - eps
            out_minus = forward().copy()

            table.data[idx] = original
            num_grad[idx] = np.sum((out_plus - out_minus) / (2 * eps) * grad_output)

        assert np.allclose(table.grad, num_grad, atol=1e-6)

    def test_backward_returns_zero_grad_for_indices(self):
        table = Tensor(np.random.randn(5, 3), requires_grad=True)
        indices = Tensor(np.array([0, 2]), requires_grad=True)
        out = Embedding.apply(table, indices)
        out.backward(np.ones_like(out.data))

        # requires_grad=True is unusual for integer indices, but backward
        # should still return an all-zero gradient rather than erroring
        assert np.array_equal(indices.grad, np.zeros(2))
