import numpy as np

from nabla.nn.transformer import TransformerBlock
from nabla.tensor import Tensor


class TestTransformerBlock:
    def test_forward_shape(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
        x = Tensor(np.random.randn(2, 6, 16))
        out = block(x)
        assert out.data.shape == (2, 6, 16)

    def test_forward_rejects_non_tensor_input(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
        try:
            block(np.random.randn(2, 6, 16))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forward_with_causal_mask_matches_shape(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
        x = Tensor(np.random.randn(2, 6, 16))
        seq_len = 6
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        out = block(x, mask=mask)
        assert out.data.shape == (2, 6, 16)

    def test_causal_mask_blocks_influence_from_future_positions(self):
        # perturbing a later position must not change an earlier position's
        # output when a causal mask is applied
        np.random.seed(0)
        block = TransformerBlock(embed_dim=8, num_heads=2, hidden_dim=16, dropout=0.0)
        seq_len = 5
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x_data = np.random.randn(1, seq_len, 8)
        x_perturbed = x_data.copy()
        x_perturbed[0, -1] = np.random.randn(8) * 100  # drastically change the last position

        out_original = block(Tensor(x_data), mask=mask)
        out_perturbed = block(Tensor(x_perturbed), mask=mask)

        # every position except the last one must be unaffected
        assert np.allclose(out_original.data[0, :-1], out_perturbed.data[0, :-1], atol=1e-4)

    def test_residual_connection_present(self):
        # zeroing every sublayer's contribution via eval-mode dropout(p=1)
        # isn't possible, so instead check output isn't independent of x:
        # a large shift in x should shift the output roughly correspondingly
        # for at least the residual path (sanity, not exact equality)
        block = TransformerBlock(embed_dim=8, num_heads=2, hidden_dim=16, dropout=0.0)
        x = Tensor(np.zeros((1, 4, 8)))
        out_zero = block(x)

        x_shifted = Tensor(np.ones((1, 4, 8)) * 1000.0)
        out_shifted = block(x_shifted)

        assert not np.allclose(out_zero.data, out_shifted.data)

    def test_parameters_include_attention_norms_and_feedforward(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
        params = block.parameters()

        expected = (
            len(block.attn.parameters())
            + len(block.norm1.parameters())
            + len(block.ff.parameters())
            + len(block.norm2.parameters())
        )
        assert len(params) == expected

    def test_backward_fills_gradients_throughout(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
        x = Tensor(np.random.randn(2, 6, 16), requires_grad=True)
        out = block(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        for param in block.parameters():
            assert param.grad is not None

    def test_train_eval_toggle_propagates_to_dropout(self):
        block = TransformerBlock(embed_dim=8, num_heads=2, hidden_dim=16, dropout=0.5)
        assert block.dropout.training is True
        block.eval()
        assert block.dropout.training is False
