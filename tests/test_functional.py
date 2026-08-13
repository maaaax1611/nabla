import numpy as np

import nabla.functional as F
from nabla.tensor import Tensor


class TestFunctional:
    def test_relu_matches_tensor_method(self):
        x_data = np.random.randn(4, 5)
        assert np.array_equal(F.relu(Tensor(x_data)).data, Tensor(x_data).relu().data)

    def test_sigmoid_matches_tensor_method(self):
        x_data = np.random.randn(4, 5)
        assert np.array_equal(F.sigmoid(Tensor(x_data)).data, Tensor(x_data).sigmoid().data)

    def test_softmax_matches_tensor_method(self):
        x_data = np.random.randn(4, 5)
        assert np.array_equal(F.softmax(Tensor(x_data), axis=-1).data, Tensor(x_data).softmax(axis=-1).data)

    def test_layer_norm_matches_tensor_method(self):
        x_data = np.random.randn(4, 5)
        gamma_data, beta_data = np.ones(5), np.zeros(5)
        expected = Tensor(x_data).layer_norm(Tensor(gamma_data), Tensor(beta_data)).data
        actual = F.layer_norm(Tensor(x_data), Tensor(gamma_data), Tensor(beta_data)).data
        assert np.array_equal(actual, expected)

    def test_matmul_supports_batched_shapes(self):
        a = Tensor(np.random.randn(2, 3, 4))
        b = Tensor(np.random.randn(2, 4, 5))
        out = F.matmul(a, b)
        assert out.data.shape == (2, 3, 5)
        assert np.allclose(out.data, np.matmul(a.data, b.data))

    def test_reshape_and_transpose_match_tensor_methods(self):
        x_data = np.arange(6).astype(float)
        assert np.array_equal(F.reshape(Tensor(x_data), (2, 3)).data, Tensor(x_data).reshape((2, 3)).data)

        y_data = np.arange(6).astype(float).reshape(2, 3)
        assert np.array_equal(F.transpose(Tensor(y_data)).data, Tensor(y_data).transpose().data)

    def test_sum_and_mean_reduce_to_scalar(self):
        x_data = np.random.randn(3, 4)
        assert np.isclose(F.sum(Tensor(x_data)).data, x_data.sum())
        assert np.isclose(F.mean(Tensor(x_data)).data, x_data.mean())

    def test_gradients_flow_through_functional_calls(self):
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        loss = F.sum(F.relu(x))
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.data.shape

    def test_dropout_scales_survivors(self):
        np.random.seed(0)
        x = Tensor(np.ones((100, 100)))
        out = F.dropout(x, p=0.3)
        survivors = out.data[out.data != 0.0]
        assert np.allclose(survivors, 1.0 / 0.7)


def naive_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V


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


class TestScaledDotProductAttention:
    def test_forward_shape(self):
        Q = Tensor(np.random.randn(2, 3, 4))
        K = Tensor(np.random.randn(2, 5, 4))
        V = Tensor(np.random.randn(2, 5, 6))
        out = F.scaled_dot_product_attention(Q, K, V)
        assert out.data.shape == (2, 3, 6)

    def test_forward_matches_naive_reference(self):
        np.random.seed(0)
        Q = Tensor(np.random.randn(2, 3, 4))
        K = Tensor(np.random.randn(2, 5, 4))
        V = Tensor(np.random.randn(2, 5, 6))
        out = F.scaled_dot_product_attention(Q, K, V)
        expected = naive_attention(Q.data, K.data, V.data)
        assert np.allclose(out.data, expected)

    def test_forward_weights_sum_to_one(self):
        # can't inspect weights directly (not returned), so check the
        # output is a convex combination of V's rows: attending uniformly
        # to a constant V should reproduce that constant exactly
        Q = Tensor(np.random.randn(2, 3, 4))
        K = Tensor(np.random.randn(2, 5, 4))
        V = Tensor(np.ones((2, 5, 6)) * 3.0)
        out = F.scaled_dot_product_attention(Q, K, V)
        assert np.allclose(out.data, 3.0)

    def test_forward_with_mask_matches_naive_reference(self):
        np.random.seed(1)
        Q = Tensor(np.random.randn(2, 3, 4))
        K = Tensor(np.random.randn(2, 5, 4))
        V = Tensor(np.random.randn(2, 5, 6))
        mask = np.zeros((2, 3, 5))
        mask[:, :, 3:] = -1e9  # block the last two key positions

        out = F.scaled_dot_product_attention(Q, K, V, mask=mask)
        expected = naive_attention(Q.data, K.data, V.data, mask=mask)
        assert np.allclose(out.data, expected)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(2)
        Q = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        K = Tensor(np.random.randn(2, 5, 4), requires_grad=True)
        V = Tensor(np.random.randn(2, 5, 6), requires_grad=True)

        def forward():
            return F.scaled_dot_product_attention(Q, K, V).data

        grad_output = np.random.randn(*forward().shape)
        out = F.scaled_dot_product_attention(Q, K, V)
        out.backward(grad_output)

        assert np.allclose(Q.grad, numerical_gradient(forward, Q, grad_output), atol=1e-6)
        assert np.allclose(K.grad, numerical_gradient(forward, K, grad_output), atol=1e-6)
        assert np.allclose(V.grad, numerical_gradient(forward, V, grad_output), atol=1e-6)

    def test_backward_matches_numerical_gradient_with_mask(self):
        np.random.seed(3)
        Q = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        K = Tensor(np.random.randn(2, 5, 4), requires_grad=True)
        V = Tensor(np.random.randn(2, 5, 6), requires_grad=True)
        mask = np.zeros((2, 3, 5))
        mask[:, :, 3:] = -1e9

        def forward():
            return F.scaled_dot_product_attention(Q, K, V, mask=mask).data

        grad_output = np.random.randn(*forward().shape)
        out = F.scaled_dot_product_attention(Q, K, V, mask=mask)
        out.backward(grad_output)

        assert np.allclose(Q.grad, numerical_gradient(forward, Q, grad_output), atol=1e-6)
        assert np.allclose(K.grad, numerical_gradient(forward, K, grad_output), atol=1e-6)
        assert np.allclose(V.grad, numerical_gradient(forward, V, grad_output), atol=1e-6)
