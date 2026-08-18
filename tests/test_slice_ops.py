import numpy as np

from nabla.ops.slice import Slice
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


class TestSlice:
    def test_forward_single_int_index(self):
        x = Tensor(np.arange(12).reshape(3, 4).astype(float))
        out = Slice.apply(x, key=1)
        assert np.array_equal(out.data, x.data[1])

    def test_forward_slice(self):
        x = Tensor(np.arange(12).reshape(3, 4).astype(float))
        out = Slice.apply(x, key=slice(0, 2))
        assert np.array_equal(out.data, x.data[0:2])

    def test_forward_tuple_key_drops_axis(self):
        # this is the ViT CLS-token pattern: pick out one fixed position
        # along an axis, dropping it from the output shape
        x = Tensor(np.arange(24).reshape(2, 3, 4).astype(float))
        out = Slice.apply(x, key=(slice(None), 0))
        assert out.data.shape == (2, 4)
        assert np.array_equal(out.data, x.data[:, 0])

    def test_forward_via_getitem(self):
        x = Tensor(np.arange(12).reshape(3, 4).astype(float))
        out = x[1:3, :2]
        assert np.array_equal(out.data, x.data[1:3, :2])

    def test_backward_scatters_gradient_to_original_shape(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        out = Slice.apply(x, key=(slice(None), 0))

        grad_output = np.ones((3,))
        out.backward(grad_output)

        assert x.grad.shape == x.data.shape
        expected = np.zeros((3, 4))
        expected[:, 0] = 1.0
        assert np.array_equal(x.grad, expected)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        key = (slice(1, 3), slice(None))

        def forward():
            return x.data[key]

        grad_output = np.random.randn(2, 5)
        out = Slice.apply(x, key=key)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_with_ellipsis_and_newaxis(self):
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        out = Slice.apply(x, key=(Ellipsis, slice(0, 2)))

        grad_output = np.ones((2, 3, 2))
        out.backward(grad_output)

        expected = np.zeros((2, 3, 4))
        expected[..., 0:2] = 1.0
        assert np.array_equal(x.grad, expected)

    def test_slicing_twice_accumulates_gradient_at_overlap(self):
        # two different Slice ops touching the same element must add up
        # (that's Tensor.backward's job, not Slice's - Slice itself never
        # sees the overlap) rather than one overwriting the other
        x = Tensor(np.zeros(3), requires_grad=True)
        a = x[0:2]
        b = x[1:3]
        out = (a.sum()) + (b.sum())
        out.backward()

        assert np.array_equal(x.grad, np.array([1.0, 2.0, 1.0]))
