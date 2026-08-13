import numpy as np

from nabla.ops.softmax import Softmax
from nabla.tensor import Tensor


def naive_softmax(x, axis):
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=axis, keepdims=True)


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


class TestSoftmax:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(4, 5))
        out = Softmax.apply(x)
        assert out.data.shape == (4, 5)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(5, 6))
        out = Softmax.apply(x)
        expected = naive_softmax(x.data, axis=-1)
        assert np.allclose(out.data, expected)

    def test_forward_sums_to_one(self):
        x = Tensor(np.random.randn(8, 4))
        out = Softmax.apply(x)
        assert np.allclose(out.data.sum(axis=-1), 1.0)

    def test_forward_is_stable_for_large_logits(self):
        # naive exp(x) would overflow for values like this
        x = Tensor(np.array([[1000.0, 1001.0, 999.0]]))
        out = Softmax.apply(x)
        assert np.all(np.isfinite(out.data))
        assert np.allclose(out.data.sum(axis=-1), 1.0)

    def test_forward_over_custom_axis(self):
        np.random.seed(1)
        x = Tensor(np.random.randn(3, 4, 5))
        out = Softmax.apply(x, axis=1)
        expected = naive_softmax(x.data, axis=1)
        assert np.allclose(out.data, expected)
        assert np.allclose(out.data.sum(axis=1), 1.0)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(2)
        x = Tensor(np.random.randn(4, 6), requires_grad=True)

        def forward():
            return Softmax().forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = Softmax.apply(x)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_over_custom_axis(self):
        np.random.seed(3)
        x = Tensor(np.random.randn(3, 4, 5), requires_grad=True)

        def forward():
            return Softmax(axis=1).forward(x)

        grad_output = np.random.randn(*forward().shape)

        out = Softmax.apply(x, axis=1)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_grad_sums_to_zero_when_grad_output_is_uniform(self):
        # a constant grad_output means the loss only depends on the (fixed)
        # sum of probs, so the gradient w.r.t. x should vanish
        np.random.seed(4)
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        out = Softmax.apply(x)
        out.backward(np.ones_like(out.data))

        assert np.allclose(x.grad, 0.0, atol=1e-10)
