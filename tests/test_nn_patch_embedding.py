import numpy as np
import pytest

from nabla.nn.patch_embedding import PatchEmbedding
from nabla.tensor import Tensor


class TestPatchEmbedding:
    def test_forward_shape(self):
        patch_embed = PatchEmbedding(img_size=28, patch_size=7, in_channels=1, embed_dim=16)
        x = Tensor(np.random.randn(3, 1, 28, 28))
        out = patch_embed(x)
        assert out.data.shape == (3, 16, 16)  # (28/7)**2 = 16 patches

    def test_num_patches_attribute(self):
        patch_embed = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=8)
        assert patch_embed.num_patches == 64

    def test_rejects_non_divisible_patch_size(self):
        with pytest.raises(ValueError):
            PatchEmbedding(img_size=28, patch_size=5, in_channels=1, embed_dim=16)

    def test_backward_fills_gradients(self):
        patch_embed = PatchEmbedding(img_size=8, patch_size=4, in_channels=1, embed_dim=6)
        x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)
        out = patch_embed(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        for param in patch_embed.parameters():
            assert param.grad is not None

    def test_patches_are_spatially_distinct(self):
        # each patch's embedding should depend only on its own pixels -
        # zeroing one patch's pixels must not change another patch's output
        patch_embed = PatchEmbedding(img_size=8, patch_size=4, in_channels=1, embed_dim=6)
        x_data = np.random.randn(1, 1, 8, 8)
        out_original = patch_embed(Tensor(x_data))

        x_zeroed = x_data.copy()
        x_zeroed[0, 0, :4, :4] = 0.0  # zero only the top-left patch
        out_zeroed = patch_embed(Tensor(x_zeroed))

        # patches other than index 0 (top-left, row-major flatten order) unaffected
        assert np.allclose(out_original.data[0, 1:], out_zeroed.data[0, 1:])
        assert not np.allclose(out_original.data[0, 0], out_zeroed.data[0, 0])
