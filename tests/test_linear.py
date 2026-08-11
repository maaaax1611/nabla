from nabla.init import xavier, zeros
from nabla.nn.linear import Linear
from nabla.tensor import Tensor
import numpy as np


class TestLinear:
    def test_forward_shape(self):
        layer = Linear(4, 8)
        x = Tensor(np.random.randn(5, 4))
        out = layer(x)
        assert out.data.shape == (5, 8)

    def test_forward_matches_manual_computation(self):
        layer = Linear(3, 2)
        x = Tensor(np.random.randn(4, 3))
        out = layer(x)
        expected = x.data @ layer.weight.data + layer.bias.data
        assert np.allclose(out.data, expected)

    def test_default_weight_init_is_he(self):
        np.random.seed(0)
        layer = Linear(256, 128)
        expected_std = np.sqrt(2.0 / 256)
        assert np.isclose(layer.weight.data.std(), expected_std, rtol=0.05)

    def test_bias_initialized_to_zero(self):
        layer = Linear(4, 8)
        assert np.all(layer.bias.data == 0)

    def test_custom_weight_init_is_used(self):
        layer = Linear(4, 8, weight_init=zeros)
        assert np.all(layer.weight.data == 0)

        layer2 = Linear(4, 8, weight_init=xavier)
        limit = np.sqrt(6.0 / (4 + 8))
        assert layer2.weight.data.min() >= -limit
        assert layer2.weight.data.max() <= limit

    def test_backward_fills_parameter_gradients(self):
        layer = Linear(3, 2)
        x = Tensor(np.random.randn(5, 3), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.data.shape
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.data.shape
        assert x.grad.shape == x.data.shape

    def test_parameters_returns_weight_and_bias(self):
        layer = Linear(3, 2)
        params = layer.parameters()
        assert len(params) == 2
        assert layer.weight in params
        assert layer.bias in params

    def test_forward_rejects_non_tensor_input(self):
        layer = Linear(3, 2)
        try:
            layer(np.random.randn(5, 3))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forward_rejects_wrong_last_dimension(self):
        layer = Linear(3, 2)
        x = Tensor(np.random.randn(5, 4))
        try:
            layer(x)
            assert False, "expected ValueError"
        except ValueError:
            pass
