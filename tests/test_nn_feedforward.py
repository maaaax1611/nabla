import numpy as np

from nabla.nn.feedforward import FeedForward
from nabla.tensor import Tensor


class TestFeedForward:
    def test_forward_shape(self):
        ff = FeedForward(embed_dim=8, hidden_dim=32)
        x = Tensor(np.random.randn(2, 5, 8))
        out = ff(x)
        assert out.data.shape == (2, 5, 8)

    def test_hidden_layer_has_expected_width(self):
        ff = FeedForward(embed_dim=8, hidden_dim=32)
        assert ff.fc1.weight.data.shape == (8, 32)
        assert ff.fc2.weight.data.shape == (32, 8)

    def test_parameters_include_both_linear_layers(self):
        ff = FeedForward(embed_dim=8, hidden_dim=32)
        params = ff.parameters()
        assert len(params) == 4
        assert ff.fc1.weight in params and ff.fc1.bias in params
        assert ff.fc2.weight in params and ff.fc2.bias in params

    def test_backward_fills_gradients(self):
        ff = FeedForward(embed_dim=8, hidden_dim=32)
        x = Tensor(np.random.randn(2, 5, 8), requires_grad=True)
        out = ff(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        assert ff.fc1.weight.grad is not None
        assert ff.fc2.weight.grad is not None
