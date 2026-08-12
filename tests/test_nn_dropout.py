import numpy as np

from nabla.nn.dropout import Dropout
from nabla.tensor import Tensor


class TestDropout:
    def test_forward_shape(self):
        layer = Dropout(p=0.5)
        x = Tensor(np.random.randn(4, 5))
        out = layer(x)
        assert out.data.shape == (4, 5)

    def test_starts_in_training_mode(self):
        layer = Dropout(p=0.5)
        assert layer.training is True

    def test_train_mode_drops_roughly_p_fraction(self):
        np.random.seed(0)
        layer = Dropout(p=0.3)
        x = Tensor(np.ones((200, 200)))
        out = layer(x)

        fraction_dropped = np.mean(out.data == 0.0)
        assert abs(fraction_dropped - 0.3) < 0.02

    def test_eval_mode_is_identity(self):
        layer = Dropout(p=0.5)
        layer.eval()
        x = Tensor(np.random.randn(4, 5))

        out = layer(x)

        assert np.array_equal(out.data, x.data)

    def test_eval_mode_returns_same_tensor_object(self):
        # eval-mode dropout is a pure pass-through, no Function should run
        layer = Dropout(p=0.5)
        layer.eval()
        x = Tensor(np.random.randn(3, 3))

        out = layer(x)

        assert out is x

    def test_train_then_eval_toggle(self):
        layer = Dropout(p=0.5)
        assert layer.training is True
        layer.eval()
        assert layer.training is False
        layer.train()
        assert layer.training is True

    def test_backward_fills_input_gradient(self):
        layer = Dropout(p=0.5)
        x = Tensor(np.random.randn(10, 10), requires_grad=True)

        out = layer(x)
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.data.shape

    def test_parameters_are_empty(self):
        layer = Dropout(p=0.5)
        assert layer.parameters() == []

    def test_forward_rejects_non_tensor_input(self):
        layer = Dropout(p=0.5)
        try:
            layer(np.random.randn(4, 5))
            assert False, "expected TypeError"
        except TypeError:
            pass
