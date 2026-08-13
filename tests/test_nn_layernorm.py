from nabla.nn.layernorm import LayerNorm
from nabla.tensor import Tensor
import numpy as np


class TestLayerNorm:
    def test_forward_shape(self):
        layer = LayerNorm(num_features=6)
        x = Tensor(np.random.randn(4, 6))
        out = layer(x)
        assert out.data.shape == (4, 6)

    def test_forward_shape_with_sequence_axis(self):
        layer = LayerNorm(num_features=8)
        x = Tensor(np.random.randn(2, 5, 8))
        out = layer(x)
        assert out.data.shape == (2, 5, 8)

    def test_initial_gamma_and_beta(self):
        layer = LayerNorm(num_features=6)
        assert np.array_equal(layer.gamma.data, np.ones(6))
        assert np.array_equal(layer.beta.data, np.zeros(6))

    def test_forward_normalizes_to_zero_mean_unit_variance(self):
        np.random.seed(0)
        layer = LayerNorm(num_features=10)
        x = Tensor(np.random.randn(6, 10) * 5 + 2)
        out = layer(x)

        per_sample_mean = out.data.mean(axis=-1)
        per_sample_std = out.data.std(axis=-1)
        assert np.allclose(per_sample_mean, 0.0, atol=1e-6)
        assert np.allclose(per_sample_std, 1.0, atol=1e-3)

    def test_parameters_returns_only_gamma_and_beta(self):
        layer = LayerNorm(num_features=6)
        params = layer.parameters()
        assert len(params) == 2
        assert layer.gamma in params
        assert layer.beta in params

    def test_backward_fills_parameter_and_input_gradients(self):
        layer = LayerNorm(num_features=6)
        x = Tensor(np.random.randn(4, 6), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert layer.gamma.grad is not None and layer.gamma.grad.shape == (6,)
        assert layer.beta.grad is not None and layer.beta.grad.shape == (6,)
        assert x.grad.shape == x.data.shape

    def test_forward_rejects_non_tensor_input(self):
        layer = LayerNorm(num_features=6)
        try:
            layer(np.random.randn(4, 6))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forward_rejects_wrong_feature_count(self):
        layer = LayerNorm(num_features=6)
        x = Tensor(np.random.randn(4, 8))
        try:
            layer(x)
            assert False, "expected ValueError"
        except ValueError:
            pass
