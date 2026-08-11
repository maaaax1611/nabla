from nabla.nn.batchnorm import BatchNorm2D
from nabla.tensor import Tensor
import numpy as np


class TestBatchNorm2D:
    def test_forward_shape(self):
        layer = BatchNorm2D(num_features=3)
        x = Tensor(np.random.randn(4, 3, 5, 5))
        out = layer(x)
        assert out.data.shape == (4, 3, 5, 5)

    def test_initial_running_stats(self):
        layer = BatchNorm2D(num_features=3)
        assert np.array_equal(layer.running_mean, np.zeros(3))
        assert np.array_equal(layer.running_var, np.ones(3))

    def test_initial_gamma_and_beta(self):
        layer = BatchNorm2D(num_features=3)
        assert np.array_equal(layer.gamma.data, np.ones(3))
        assert np.array_equal(layer.beta.data, np.zeros(3))

    def test_starts_in_training_mode(self):
        layer = BatchNorm2D(num_features=3)
        assert layer.training is True

    def test_train_mode_normalizes_to_zero_mean_unit_variance(self):
        np.random.seed(0)
        layer = BatchNorm2D(num_features=3)
        x = Tensor(np.random.randn(8, 3, 4, 4) * 5 + 2)
        out = layer(x)

        per_channel_mean = out.data.mean(axis=(0, 2, 3))
        per_channel_std = out.data.std(axis=(0, 2, 3))
        assert np.allclose(per_channel_mean, 0.0, atol=1e-6)
        assert np.allclose(per_channel_std, 1.0, atol=1e-3)

    def test_train_mode_updates_running_stats_with_momentum(self):
        np.random.seed(1)
        layer = BatchNorm2D(num_features=3, momentum=0.1)
        x = Tensor(np.random.randn(8, 3, 4, 4))

        layer(x)

        expected_mean = 0.1 * x.data.mean(axis=(0, 2, 3))
        expected_var = 0.1 * x.data.var(axis=(0, 2, 3))
        assert np.allclose(layer.running_mean, expected_mean)
        assert np.allclose(layer.running_var, expected_var + 0.9)  # running_var started at 1

    def test_eval_mode_uses_running_stats_not_batch_stats(self):
        layer = BatchNorm2D(num_features=2)
        # fake "trained" running stats, very different from the eval input's own stats
        layer.running_mean = np.array([10.0, -10.0])
        layer.running_var = np.array([4.0, 9.0])
        layer.eval()

        x = Tensor(np.random.randn(1, 2, 3, 3))
        out = layer(x)

        expected = (
            layer.gamma.data.reshape(1, -1, 1, 1)
            * (x.data - layer.running_mean.reshape(1, -1, 1, 1))
            / np.sqrt(layer.running_var.reshape(1, -1, 1, 1) + layer.eps)
            + layer.beta.data.reshape(1, -1, 1, 1)
        )
        assert np.allclose(out.data, expected)

    def test_eval_mode_does_not_change_running_stats(self):
        layer = BatchNorm2D(num_features=2)
        layer.eval()
        before_mean, before_var = layer.running_mean.copy(), layer.running_var.copy()

        layer(Tensor(np.random.randn(4, 2, 3, 3)))

        assert np.array_equal(layer.running_mean, before_mean)
        assert np.array_equal(layer.running_var, before_var)

    def test_train_then_eval_toggle(self):
        layer = BatchNorm2D(num_features=2)
        assert layer.training is True
        layer.eval()
        assert layer.training is False
        layer.train()
        assert layer.training is True

    def test_parameters_returns_only_gamma_and_beta(self):
        layer = BatchNorm2D(num_features=3)
        params = layer.parameters()
        assert len(params) == 2
        assert layer.gamma in params
        assert layer.beta in params

    def test_backward_fills_parameter_and_input_gradients(self):
        layer = BatchNorm2D(num_features=3)
        x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert layer.gamma.grad is not None and layer.gamma.grad.shape == (3,)
        assert layer.beta.grad is not None and layer.beta.grad.shape == (3,)
        assert x.grad.shape == x.data.shape

    def test_backward_matches_numerical_gradient_in_eval_mode(self):
        np.random.seed(2)
        layer = BatchNorm2D(num_features=2)
        layer.running_mean = np.array([0.5, -0.5])
        layer.running_var = np.array([2.0, 3.0])
        layer.eval()

        x = Tensor(np.random.randn(2, 2, 3, 3), requires_grad=True)

        def forward():
            return layer.forward(x).data

        grad_output = np.random.randn(*forward().shape)
        out = layer(x)
        out.backward(grad_output)

        eps = 1e-5
        num_grad = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = x.data[idx]
            x.data[idx] = original + eps
            out_plus = forward().copy()
            x.data[idx] = original - eps
            out_minus = forward().copy()
            x.data[idx] = original
            num_grad[idx] = np.sum((out_plus - out_minus) / (2 * eps) * grad_output)

        assert np.allclose(x.grad, num_grad, atol=1e-6)

    def test_forward_rejects_non_tensor_input(self):
        layer = BatchNorm2D(num_features=3)
        try:
            layer(np.random.randn(4, 3, 5, 5))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forward_rejects_wrong_channel_count(self):
        layer = BatchNorm2D(num_features=3)
        x = Tensor(np.random.randn(4, 5, 5, 5))
        try:
            layer(x)
            assert False, "expected ValueError"
        except ValueError:
            pass
