import numpy as np

from nabla.ops.concat import Concat
from nabla.tensor import Tensor


def numerical_gradient(f, t, grad_output, eps=1e-5):
    """Central-difference gradient of sum(f() * grad_output) w.r.t. t.data."""
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = t.data[idx]

        t.data[idx] = original + eps
        out_plus = f().copy()

        t.data[idx] = original - eps
        out_minus = f().copy()

        t.data[idx] = original
        grad[idx] = np.sum((out_plus - out_minus) / (2 * eps) * grad_output)
    return grad


class TestConcat:
    def test_forward_two_tensors(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = Concat.apply(a, b, axis=1)
        expected = np.concatenate([a.data, b.data], axis=1)
        assert np.array_equal(c.data, expected)

    def test_forward_three_tensors_axis_0(self):
        a = Tensor(np.ones((1, 4)))
        b = Tensor(np.ones((2, 4)) * 2)
        c = Tensor(np.ones((3, 4)) * 3)
        out = Concat.apply(a, b, c, axis=0)
        assert out.data.shape == (6, 4)
        expected = np.concatenate([a.data, b.data, c.data], axis=0)
        assert np.array_equal(out.data, expected)

    def test_backward_splits_gradient_back_to_each_input(self):
        np.random.seed(0)
        a = Tensor(np.random.randn(2, 3), requires_grad=True)
        b = Tensor(np.random.randn(2, 5), requires_grad=True)
        c = Tensor(np.random.randn(2, 2), requires_grad=True)

        out = Concat.apply(a, b, c, axis=1)
        grad_output = np.random.randn(2, 10)
        out.backward(grad_output)

        assert np.array_equal(a.grad, grad_output[:, :3])
        assert np.array_equal(b.grad, grad_output[:, 3:8])
        assert np.array_equal(c.grad, grad_output[:, 8:])

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(1)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 2), requires_grad=True)

        def forward():
            return np.concatenate([a.data, b.data], axis=1)

        grad_output = np.random.randn(3, 6)
        out = Concat.apply(a, b, axis=1)
        out.backward(grad_output)

        assert np.allclose(a.grad, numerical_gradient(forward, a, grad_output), atol=1e-6)
        assert np.allclose(b.grad, numerical_gradient(forward, b, grad_output), atol=1e-6)

    def test_concat_then_split_shapes_round_trip(self):
        # each piece's gradient shape must match that piece's own shape,
        # not get mixed up with a neighbor's
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = Tensor(np.random.randn(2, 3, 7), requires_grad=True)
        out = Concat.apply(a, b, axis=2)
        assert out.data.shape == (2, 3, 11)

        out.backward(np.ones((2, 3, 11)))
        assert a.grad.shape == a.data.shape
        assert b.grad.shape == b.data.shape
