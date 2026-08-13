import numpy as np

from nabla.nn.softmax import Softmax
from nabla.tensor import Tensor


class TestSoftmax:
    def test_forward_shape(self):
        layer = Softmax()
        x = Tensor(np.random.randn(4, 5))
        out = layer(x)
        assert out.data.shape == (4, 5)

    def test_forward_sums_to_one(self):
        layer = Softmax()
        x = Tensor(np.random.randn(4, 5))
        out = layer(x)
        assert np.allclose(out.data.sum(axis=-1), 1.0)

    def test_forward_over_custom_axis(self):
        layer = Softmax(axis=1)
        x = Tensor(np.random.randn(3, 4, 5))
        out = layer(x)
        assert np.allclose(out.data.sum(axis=1), 1.0)

    def test_parameters_are_empty(self):
        layer = Softmax()
        assert layer.parameters() == []

    def test_backward_fills_input_gradient(self):
        layer = Softmax()
        x = Tensor(np.random.randn(4, 5), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.data.shape

    def test_forward_rejects_non_tensor_input(self):
        layer = Softmax()
        try:
            layer(np.random.randn(4, 5))
            assert False, "expected TypeError"
        except TypeError:
            pass
