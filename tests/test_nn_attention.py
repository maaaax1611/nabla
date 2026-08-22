import numpy as np

from nabla.nn.attention import Attention
from nabla.tensor import Tensor


def numerical_gradient(f, t, grad_output, eps=1e-5):
    """Central-difference gradient of sum(f() * grad_output) w.r.t. t.data.

    Perturbs (and evaluates) in float64 regardless of t.data's own dtype:
    module parameters default to float32 (see src/nabla/init.py), and a
    1e-5 perturbation written into a float32 array loses enough precision
    on the round-trip that the resulting finite-difference estimate picks
    up a ~0.1% relative error - comfortably outside this test's atol. The
    production forward/backward pass still runs at whatever dtype the
    model actually uses; only this reference check needs the extra
    precision.
    """
    original_dtype = t.data.dtype
    t.data = t.data.astype(np.float64)
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = t.data[idx]

        t.data[idx] = original + eps
        out_plus = f().copy()

        t.data[idx] = original - eps
        out_minus = f().copy()

        t.data[idx] = original
        grad[idx] = np.sum((out_plus - out_minus) / (2 * eps) * grad_output)
    t.data = t.data.astype(original_dtype)
    return grad


class TestAttention:
    def test_forward_shape_self_attention(self):
        layer = Attention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(3, 5, 8))
        out = layer(x)
        assert out.data.shape == (3, 5, 8)

    def test_forward_shape_single_head(self):
        layer = Attention(embed_dim=8, num_heads=1)
        x = Tensor(np.random.randn(3, 5, 8))
        out = layer(x)
        assert out.data.shape == (3, 5, 8)

    def test_forward_shape_cross_attention(self):
        # query and key/value can have different sequence lengths
        layer = Attention(embed_dim=8, num_heads=2)
        query = Tensor(np.random.randn(3, 5, 8))
        key_value = Tensor(np.random.randn(3, 7, 8))
        out = layer(query, key_value, key_value)
        assert out.data.shape == (3, 5, 8)

    def test_init_rejects_embed_dim_not_divisible_by_num_heads(self):
        try:
            Attention(embed_dim=8, num_heads=3)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_forward_rejects_non_tensor_query(self):
        layer = Attention(embed_dim=8, num_heads=2)
        try:
            layer(np.random.randn(3, 5, 8))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_parameters_include_all_four_projections(self):
        layer = Attention(embed_dim=8, num_heads=2)
        params = layer.parameters()
        # weight + bias for each of q_proj, k_proj, v_proj, out_proj
        assert len(params) == 8
        for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            assert proj.weight in params
            assert proj.bias in params

    def test_split_and_merge_heads_are_inverses(self):
        layer = Attention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(3, 5, 8))
        merged = layer._merge_heads(layer._split_heads(x))
        assert np.allclose(merged.data, x.data)

    def test_backward_fills_gradients_for_input_and_all_projections(self):
        layer = Attention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(3, 5, 8), requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            assert proj.weight.grad is not None and proj.weight.grad.shape == proj.weight.data.shape
            assert proj.bias.grad is not None and proj.bias.grad.shape == proj.bias.data.shape

    def test_backward_matches_numerical_gradient(self):
        np.random.seed(0)
        layer = Attention(embed_dim=6, num_heads=2)
        x = Tensor(np.random.randn(2, 4, 6), requires_grad=True)

        def forward():
            return layer(x).data

        grad_output = np.random.randn(*forward().shape)
        out = layer(x)
        out.backward(grad_output)

        assert np.allclose(x.grad, numerical_gradient(forward, x, grad_output), atol=1e-5)
        assert np.allclose(
            layer.q_proj.weight.grad,
            numerical_gradient(forward, layer.q_proj.weight, grad_output),
            atol=1e-5,
        )

    def test_forward_with_mask_blocks_attention_to_masked_positions(self):
        # if a key position is fully masked out for every query, its value
        # should have no influence on the output at all
        np.random.seed(1)
        layer = Attention(embed_dim=4, num_heads=1)
        x = Tensor(np.random.randn(1, 3, 4))
        x_perturbed = x.data.copy()
        x_perturbed[0, 2] = np.random.randn(4) * 100  # drastically change key/value 2

        mask = np.zeros((1, 3, 3))
        mask[:, :, 2] = -1e9  # block every query from attending to key 2

        out_original = layer(x, x, x, mask=mask)
        out_perturbed = layer(Tensor(x.data), Tensor(x_perturbed), Tensor(x_perturbed), mask=mask)

        assert np.allclose(out_original.data, out_perturbed.data, atol=1e-4)
