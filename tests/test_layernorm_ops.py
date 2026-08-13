from nabla.tensor import Tensor
from nabla.ops.layernorm import LayerNorm
import numpy as np


def naive_layernorm(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


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


class TestLayerNorm:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(4, 6))
        gamma = Tensor(np.ones(6))
        beta = Tensor(np.zeros(6))

        out = LayerNorm.apply(x, gamma, beta)

        assert out.data.shape == (4, 6)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(5, 8))
        gamma = Tensor(np.random.randn(8))
        beta = Tensor(np.random.randn(8))

        out = LayerNorm.apply(x, gamma, beta)
        expected = naive_layernorm(x.data, gamma.data, beta.data, eps=1e-5)

        assert np.allclose(out.data, expected)

    def test_forward_normalizes_to_zero_mean_unit_variance(self):
        # with gamma=1, beta=0 the output should be standard-normalized per
        # sample, over the last axis
        np.random.seed(1)
        x = Tensor(np.random.randn(6, 10) * 5 + 2)
        gamma = Tensor(np.ones(10))
        beta = Tensor(np.zeros(10))

        out = LayerNorm.apply(x, gamma, beta)

        per_sample_mean = out.data.mean(axis=-1)
        per_sample_std = out.data.std(axis=-1)
        assert np.allclose(per_sample_mean, 0.0, atol=1e-6)
        assert np.allclose(per_sample_std, 1.0, atol=1e-3)

    def test_forward_samples_are_independent(self):
        # a constant offset on one sample shouldn't leak into another
        # sample's statistics (unlike BatchNorm2D, which mixes across the
        # batch)
        np.random.seed(2)
        x_data = np.random.randn(4, 5)
        x_data[1] += 100.0
        x = Tensor(x_data)
        gamma = Tensor(np.ones(5))
        beta = Tensor(np.zeros(5))

        out = LayerNorm.apply(x, gamma, beta)
        expected = naive_layernorm(x.data, gamma.data, beta.data, eps=1e-5)

        assert np.allclose(out.data, expected)

    def test_forward_supports_leading_batch_and_sequence_axes(self):
        # (batch, seq_len, features) - the shape a Transformer would use
        np.random.seed(3)
        x = Tensor(np.random.randn(2, 3, 4))
        gamma = Tensor(np.random.randn(4))
        beta = Tensor(np.random.randn(4))

        out = LayerNorm.apply(x, gamma, beta)
        expected = naive_layernorm(x.data, gamma.data, beta.data, eps=1e-5)

        assert out.data.shape == (2, 3, 4)
        assert np.allclose(out.data, expected)

    def test_backward_grad_beta_is_sum_over_leading_axes(self):
        x = Tensor(np.random.randn(5, 6), requires_grad=True)
        gamma = Tensor(np.random.randn(6), requires_grad=True)
        beta = Tensor(np.random.randn(6), requires_grad=True)

        out = LayerNorm.apply(x, gamma, beta)
        grad_output = np.ones_like(out.data)
        out.backward(grad_output)

        expected_grad_beta = grad_output.sum(axis=0)
        assert np.allclose(beta.grad, expected_grad_beta)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(4)
        x = Tensor(np.random.randn(5, 6), requires_grad=True)
        gamma = Tensor(np.random.randn(6), requires_grad=True)
        beta = Tensor(np.random.randn(6), requires_grad=True)

        def forward():
            return LayerNorm().forward(x, gamma, beta)

        grad_output = np.random.randn(*forward().shape)

        out = LayerNorm.apply(x, gamma, beta)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(gamma.grad, numerical_gradient(forward, gamma, grad_output), atol=1e-6)
        assert np.allclose(beta.grad, numerical_gradient(forward, beta, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_with_sequence_axis(self):
        np.random.seed(5)
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        gamma = Tensor(np.random.randn(4), requires_grad=True)
        beta = Tensor(np.random.randn(4), requires_grad=True)

        def forward():
            return LayerNorm().forward(x, gamma, beta)

        grad_output = np.random.randn(*forward().shape)

        out = LayerNorm.apply(x, gamma, beta)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(gamma.grad, numerical_gradient(forward, gamma, grad_output), atol=1e-6)
        assert np.allclose(beta.grad, numerical_gradient(forward, beta, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_single_feature(self):
        # edge case: a single feature per sample, var is always 0
        np.random.seed(6)
        x = Tensor(np.random.randn(4, 1), requires_grad=True)
        gamma = Tensor(np.random.randn(1), requires_grad=True)
        beta = Tensor(np.random.randn(1), requires_grad=True)

        def forward():
            return LayerNorm().forward(x, gamma, beta)

        grad_output = np.random.randn(*forward().shape)

        out = LayerNorm.apply(x, gamma, beta)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)
        assert np.allclose(gamma.grad, numerical_gradient(forward, gamma, grad_output), atol=1e-6)
        assert np.allclose(beta.grad, numerical_gradient(forward, beta, grad_output), atol=1e-6)
