import numpy as np

from nabla.nn.vit import VisionTransformer
from nabla.tensor import Tensor


def make_vit(**overrides):
    defaults = dict(
        img_size=8,
        patch_size=4,
        in_channels=1,
        num_classes=5,
        embed_dim=12,
        num_heads=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    defaults.update(overrides)
    return VisionTransformer(**defaults)


class TestVisionTransformer:
    def test_forward_shape(self):
        model = make_vit()
        x = Tensor(np.random.randn(3, 1, 8, 8))
        out = model(x)
        assert out.data.shape == (3, 5)

    def test_forward_rgb_input(self):
        model = make_vit(img_size=16, patch_size=4, in_channels=3)
        x = Tensor(np.random.randn(2, 3, 16, 16))
        out = model(x)
        assert out.data.shape == (2, 5)

    def test_backward_fills_gradients_throughout(self):
        model = make_vit()
        x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        for param in model.parameters():
            assert param.grad is not None, "every parameter should receive a gradient"

    def test_parameters_include_cls_token_and_pos_embedding(self):
        model = make_vit()
        params = model.parameters()
        assert any(p is model.cls_token for p in params)
        assert any(p is model.pos_embedding for p in params)

    def test_different_images_produce_different_logits(self):
        model = make_vit()
        x1 = Tensor(np.random.randn(1, 1, 8, 8))
        x2 = Tensor(np.random.randn(1, 1, 8, 8))
        out1 = model(x1)
        out2 = model(x2)
        assert not np.allclose(out1.data, out2.data)

    def test_train_eval_toggle_propagates_to_blocks(self):
        model = make_vit(dropout=0.5)
        assert model.blocks[0].dropout.training is True
        model.eval()
        assert model.blocks[0].dropout.training is False
