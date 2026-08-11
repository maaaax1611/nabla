from nabla.tensor import Tensor
from nabla.ops.batchnorm import BatchNorm2D
import numpy as np


def naive_batchnorm2d(x, gamma, beta, eps):
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma.reshape(1, -1, 1, 1) * x_hat + beta.reshape(1, -1, 1, 1)


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


class TestBatchNorm2D:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(2, 3, 4, 4))
        gamma = Tensor(np.ones(3))
        beta = Tensor(np.zeros(3))

        out = BatchNorm2D.apply(x, gamma, beta)

        assert out.data.shape == (2, 3, 4, 4)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(4, 3, 5, 5))
        gamma = Tensor(np.random.randn(3))
        beta = Tensor(np.random.randn(3))

        out = BatchNorm2D.apply(x, gamma, beta)
        expected = naive_batchnorm2d(x.data, gamma.data, beta.data, eps=1e-5)

        assert np.allclose(out.data, expected)

    def test_forward_normalizes_to_zero_mean_unit_variance(self):
        # with gamma=1, beta=0 the output should be a standard-normalized
        # version of x, per channel, over (batch, H, W).
        np.random.seed(1)
        x = Tensor(np.random.randn(8, 3, 4, 4) * 5 + 2)
        gamma = Tensor(np.ones(3))
        beta = Tensor(np.zeros(3))

        out = BatchNorm2D.apply(x, gamma, beta)

        per_channel_mean = out.data.mean(axis=(0, 2, 3))
        per_channel_std = out.data.std(axis=(0, 2, 3))
        assert np.allclose(per_channel_mean, 0.0, atol=1e-6)
        assert np.allclose(per_channel_std, 1.0, atol=1e-3)

    def test_forward_channels_are_independent(self):
        # a constant offset per channel should not leak into other channels'
        # statistics.
        np.random.seed(2)
        x_data = np.random.randn(4, 2, 3, 3)
        x_data[:, 1] += 100.0  # shift only the second channel
        x = Tensor(x_data)
        gamma = Tensor(np.ones(2))
        beta = Tensor(np.zeros(2))

        out = BatchNorm2D.apply(x, gamma, beta)
        expected = naive_batchnorm2d(x.data, gamma.data, beta.data, eps=1e-5)

        assert np.allclose(out.data, expected)

    def test_backward_grad_beta_is_sum_over_batch_and_spatial_dims(self):
        x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
        gamma = Tensor(np.random.randn(3), requires_grad=True)
        beta = Tensor(np.random.randn(3), requires_grad=True)

        out = BatchNorm2D.apply(x, gamma, beta)
        grad_output = np.ones_like(out.data)
        out.backward(grad_output)

        expected_grad_beta = grad_output.sum(axis=(0, 2, 3))
        assert np.allclose(beta.grad, expected_grad_beta)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(3)
        x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
        gamma = Tensor(np.random.randn(3), requires_grad=True)
        beta = Tensor(np.random.randn(3), requires_grad=True)

        def forward():
            return BatchNorm2D().forward(x, gamma, beta)

        grad_output = np.random.randn(*forward().shape)

        out = BatchNorm2D.apply(x, gamma, beta)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(gamma.grad, numerical_gradient(forward, gamma, grad_output), atol=1e-6)
        assert np.allclose(beta.grad, numerical_gradient(forward, beta, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_single_batch_element(self):
        # edge case: batch=1, so variance is computed over H*W only per channel
        np.random.seed(4)
        x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
        gamma = Tensor(np.random.randn(2), requires_grad=True)
        beta = Tensor(np.random.randn(2), requires_grad=True)

        def forward():
            return BatchNorm2D().forward(x, gamma, beta)

        grad_output = np.random.randn(*forward().shape)

        out = BatchNorm2D.apply(x, gamma, beta)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(gamma.grad, numerical_gradient(forward, gamma, grad_output), atol=1e-6)
        assert np.allclose(beta.grad, numerical_gradient(forward, beta, grad_output), atol=1e-6)
