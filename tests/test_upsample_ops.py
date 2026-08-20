import numpy as np

from nabla.nn.upsample import Upsample as UpsampleModule
from nabla.ops.upsample import Upsample
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


class TestUpsample:
    def test_forward_duplicates_each_pixel_into_a_block(self):
        x = Tensor(np.array([[[[0.0, 1.0], [2.0, 3.0]]]]))  # (1, 1, 2, 2)
        out = Upsample.apply(x, scale=2)

        expected = np.array(
            [
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [2.0, 2.0, 3.0, 3.0],
                        [2.0, 2.0, 3.0, 3.0],
                    ]
                ]
            ]
        )
        assert np.array_equal(out.data, expected)

    def test_forward_shape(self):
        x = Tensor(np.random.randn(2, 3, 4, 5))
        out = Upsample.apply(x, scale=3)
        assert out.data.shape == (2, 3, 12, 15)

    def test_scale_one_is_a_no_op(self):
        x = Tensor(np.random.randn(2, 3, 4, 5))
        out = Upsample.apply(x, scale=1)
        assert np.array_equal(out.data, x.data)

    def test_backward_sums_each_block_back_to_one_pixel(self):
        x = Tensor(np.zeros((1, 1, 2, 2)), requires_grad=True)
        out = Upsample.apply(x, scale=2)

        grad_output = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
        out.backward(grad_output)

        # block (0,0) = [[0,1],[4,5]] -> 10, block (0,1) = [[2,3],[6,7]] -> 18, etc.
        expected = np.array([[[[10.0, 18.0], [42.0, 50.0]]]])
        assert np.array_equal(x.grad, expected)

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(0)
        x = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
        scale = 2

        def forward():
            return x.data.repeat(scale, axis=2).repeat(scale, axis=3)

        grad_output = np.random.randn(2, 3, 8, 10)
        out = Upsample.apply(x, scale=scale)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-6)

    def test_backward_grad_shape_matches_input_shape(self):
        x = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
        out = Upsample.apply(x, scale=3)
        out.backward(np.ones((2, 3, 12, 15)))
        assert x.grad.shape == x.data.shape

    def test_via_tensor_method(self):
        x = Tensor(np.random.randn(1, 1, 2, 2))
        out = x.upsample(scale=2)
        assert out.data.shape == (1, 1, 4, 4)

    def test_nn_module_forward(self):
        layer = UpsampleModule(scale=2)
        x = Tensor(np.random.randn(1, 2, 3, 3))
        out = layer(x)
        assert out.data.shape == (1, 2, 6, 6)

    def test_nn_module_has_no_parameters(self):
        layer = UpsampleModule(scale=2)
        assert list(layer.parameters()) == []
