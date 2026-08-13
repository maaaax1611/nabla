import numpy as np

from nabla.nn.embedding import Embedding
from nabla.tensor import Tensor


class TestEmbedding:
    def test_forward_shape(self):
        layer = Embedding(vocab_size=10, embed_dim=4)
        indices = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        out = layer(indices)
        assert out.data.shape == (2, 3, 4)

    def test_forward_matches_weight_rows(self):
        layer = Embedding(vocab_size=10, embed_dim=4)
        indices = Tensor(np.array([0, 5, 9]))
        out = layer(indices)
        assert np.array_equal(out.data, layer.weight.data[[0, 5, 9]])

    def test_weight_shape_and_requires_grad(self):
        layer = Embedding(vocab_size=10, embed_dim=4)
        assert layer.weight.data.shape == (10, 4)
        assert layer.weight.requires_grad is True

    def test_parameters_returns_only_weight(self):
        layer = Embedding(vocab_size=10, embed_dim=4)
        params = layer.parameters()
        assert len(params) == 1
        assert layer.weight in params

    def test_backward_accumulates_gradient_for_repeated_indices(self):
        layer = Embedding(vocab_size=5, embed_dim=3)
        indices = Tensor(np.array([2, 2]))
        out = layer(indices)
        out.backward(np.ones_like(out.data))

        assert np.array_equal(layer.weight.grad[2], np.array([2.0, 2.0, 2.0]))
        assert np.array_equal(layer.weight.grad[0], np.zeros(3))

    def test_forward_rejects_non_tensor_input(self):
        layer = Embedding(vocab_size=10, embed_dim=4)
        try:
            layer(np.array([1, 2, 3]))
            assert False, "expected TypeError"
        except TypeError:
            pass
