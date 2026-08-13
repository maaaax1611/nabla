import numpy as np

from nabla.nn.positional_encoding import PositionalEncoding, _sinusoidal_table
from nabla.tensor import Tensor


class TestSinusoidalTable:
    def test_shape(self):
        table = _sinusoidal_table(max_len=20, embed_dim=8)
        assert table.shape == (20, 8)

    def test_values_are_bounded(self):
        table = _sinusoidal_table(max_len=20, embed_dim=8)
        assert np.all(np.abs(table) <= 1.0)

    def test_position_zero_alternates_zero_and_one(self):
        # sin(0) = 0, cos(0) = 1 for every frequency at position 0
        table = _sinusoidal_table(max_len=5, embed_dim=6)
        assert np.allclose(table[0, 0::2], 0.0)
        assert np.allclose(table[0, 1::2], 1.0)

    def test_each_position_gets_a_distinct_encoding(self):
        table = _sinusoidal_table(max_len=20, embed_dim=8)
        for i in range(20):
            for j in range(i + 1, 20):
                assert not np.allclose(table[i], table[j])


class TestPositionalEncoding:
    def test_forward_shape(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        x = Tensor(np.random.randn(2, 5, 8))
        out = pe(x)
        assert out.data.shape == (2, 5, 8)

    def test_forward_adds_same_table_slice_to_every_batch_element(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        x = Tensor(np.zeros((3, 5, 8)))
        out = pe(x)
        for b in range(3):
            assert np.allclose(out.data[b], pe.table[:5])

    def test_forward_uses_only_a_prefix_of_the_table(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        x = Tensor(np.zeros((1, 5, 8)))
        out = pe(x)
        assert np.allclose(out.data[0], pe.table[:5])

    def test_forward_rejects_sequence_longer_than_max_len(self):
        pe = PositionalEncoding(embed_dim=8, max_len=10)
        x = Tensor(np.zeros((1, 11, 8)))
        try:
            pe(x)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_forward_rejects_wrong_embed_dim(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        x = Tensor(np.zeros((1, 5, 4)))
        try:
            pe(x)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_forward_rejects_non_tensor_input(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        try:
            pe(np.zeros((1, 5, 8)))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_table_is_not_a_trainable_parameter(self):
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        assert pe.parameters() == []

    def test_backward_passes_gradient_through_unchanged(self):
        # adding a constant doesn't change the gradient at all
        pe = PositionalEncoding(embed_dim=8, max_len=50)
        x = Tensor(np.random.randn(2, 5, 8), requires_grad=True)
        out = pe(x)
        grad_output = np.random.randn(*out.data.shape)
        out.backward(grad_output)
        assert np.array_equal(x.grad, grad_output)
