import numpy as np

from nabla.ops.dropout import Dropout
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


class TestDropout:
    def test_forward_shape(self):
        x = Tensor(np.random.randn(4, 5))
        out = Dropout.apply(x, p=0.5)
        assert out.data.shape == (4, 5)

    def test_forward_p_zero_leaves_x_unchanged(self):
        x = Tensor(np.random.randn(4, 5))
        out = Dropout.apply(x, p=0.0)
        assert np.allclose(out.data, x.data)

    def test_forward_drops_roughly_p_fraction(self):
        np.random.seed(0)
        x = Tensor(np.ones((200, 200)))
        out = Dropout.apply(x, p=0.3)

        fraction_dropped = np.mean(out.data == 0.0)
        assert abs(fraction_dropped - 0.3) < 0.02

    def test_forward_survivors_are_scaled_by_inverse_keep_prob(self):
        np.random.seed(1)
        p = 0.4
        x = Tensor(np.ones((100, 100)))
        out = Dropout.apply(x, p=p)

        survivors = out.data[out.data != 0.0]
        assert np.allclose(survivors, 1.0 / (1.0 - p))

    def test_backward_zero_grad_at_dropped_positions(self):
        np.random.seed(2)
        x = Tensor(np.random.randn(50, 50), requires_grad=True)
        out = Dropout.apply(x, p=0.5)

        grad_output = np.ones_like(out.data)
        out.backward(grad_output)

        dropped = out.data == 0.0
        assert np.allclose(x.grad[dropped], 0.0)
        assert np.allclose(x.grad[~dropped], 1.0 / (1.0 - 0.5))

    def test_backward_matches_numerical_gradient_with_fixed_mask(self):
        # forward is stochastic, so the numerical gradient check reuses the
        # exact mask sampled during the real forward pass instead of
        # resampling it on every eps-perturbed call.
        np.random.seed(3)
        x = Tensor(np.random.randn(6, 7), requires_grad=True)

        out = Dropout.apply(x, p=0.3)
        mask = out._ctx.mask

        def forward_fixed_mask():
            return x.data * mask

        grad_output = np.random.randn(*out.data.shape)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward_fixed_mask, x, grad_output))
