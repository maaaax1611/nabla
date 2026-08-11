from nabla.init import xavier, zeros
from nabla.nn.conv import Conv2D
from nabla.tensor import Tensor
import numpy as np


class TestConv2D:
    def test_forward_shape(self):
        layer = Conv2D(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 10, 10))
        out = layer(x)
        assert out.data.shape == (2, 8, 10, 10)

    def test_forward_shape_with_stride(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=2)
        x = Tensor(np.random.randn(2, 3, 9, 9))
        out = layer(x)
        assert out.data.shape == (2, 4, 4, 4)

    def test_forward_matches_manual_computation(self):
        layer = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 2, 5, 5))
        out = layer(x)
        expected = x.conv2d(layer.weight, layer.bias, stride=layer.stride, padding=layer.padding)
        assert np.allclose(out.data, expected.data)

    def test_default_weight_init_is_he(self):
        np.random.seed(0)
        layer = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
        fan_in = 8 * 3 * 3
        expected_std = np.sqrt(2.0 / fan_in)
        assert np.isclose(layer.weight.data.std(), expected_std, rtol=0.1)

    def test_bias_initialized_to_zero(self):
        layer = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        assert np.all(layer.bias.data == 0)

    def test_custom_weight_init_is_used(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3, weight_init=zeros)
        assert np.all(layer.weight.data == 0)

        layer2 = Conv2D(in_channels=3, out_channels=4, kernel_size=3, weight_init=xavier)
        fan_in, fan_out = 3 * 3 * 3, 4 * 3 * 3
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert layer2.weight.data.min() >= -limit
        assert layer2.weight.data.max() <= limit

    def test_backward_fills_parameter_gradients(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 6, 6), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.data.shape
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.data.shape
        assert x.grad.shape == x.data.shape

    def test_parameters_returns_weight_and_bias(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3)
        params = layer.parameters()
        assert len(params) == 2
        assert layer.weight in params
        assert layer.bias in params

    def test_forward_rejects_non_tensor_input(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3)
        try:
            layer(np.random.randn(2, 3, 6, 6))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forward_rejects_wrong_channel_count(self):
        layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3)
        x = Tensor(np.random.randn(2, 5, 6, 6))
        try:
            layer(x)
            assert False, "expected ValueError"
        except ValueError:
            pass
