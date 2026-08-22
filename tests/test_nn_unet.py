import numpy as np

from nabla.nn.unet import DoubleConv, UNet
from nabla.tensor import Tensor


def make_unet(**overrides):
    defaults = dict(in_channels=2, out_channels=1, features=(4, 8))
    defaults.update(overrides)
    return UNet(**defaults)


class TestDoubleConv:
    def test_forward_shape_preserves_h_and_w(self):
        block = DoubleConv(in_channels=3, out_channels=8)
        x = Tensor(np.random.randn(2, 3, 16, 16))
        out = block(x)
        assert out.data.shape == (2, 8, 16, 16)

    def test_forward_output_is_non_negative_after_relu(self):
        block = DoubleConv(in_channels=3, out_channels=8)
        x = Tensor(np.random.randn(2, 3, 16, 16))
        out = block(x)
        assert (out.data >= 0).all()


class TestUNet:
    def test_forward_shape_matches_input_h_and_w(self):
        # a 2-level U-Net halves resolution twice, so H/W must be
        # divisible by 2**2 = 4 for the encoder/decoder shapes to line up
        model = make_unet()
        x = Tensor(np.random.randn(2, 2, 32, 32))
        out = model(x)
        assert out.data.shape == (2, 1, 32, 32)

    def test_forward_output_is_a_valid_probability_map(self):
        # final activation is sigmoid, so every pixel must be in [0, 1] -
        # ready to feed directly into DiceLoss
        model = make_unet()
        x = Tensor(np.random.randn(2, 2, 32, 32))
        out = model(x)
        assert (out.data >= 0).all() and (out.data <= 1).all()

    def test_forward_rejects_non_tensor_input(self):
        model = make_unet()
        try:
            model(np.random.randn(1, 2, 32, 32))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_multi_channel_output(self):
        model = make_unet(out_channels=3)
        x = Tensor(np.random.randn(1, 2, 32, 32))
        out = model(x)
        assert out.data.shape == (1, 3, 32, 32)

    def test_deeper_feature_stack(self):
        # 3 pooling levels -> H/W must be divisible by 8
        model = make_unet(features=(4, 8, 16))
        x = Tensor(np.random.randn(1, 2, 32, 32))
        out = model(x)
        assert out.data.shape == (1, 1, 32, 32)

    def test_backward_fills_gradients_throughout(self):
        model = make_unet()
        x = Tensor(np.random.randn(2, 2, 32, 32), requires_grad=True)
        out = model(x)
        out.sum().backward()

        assert x.grad is not None and x.grad.shape == x.data.shape
        for param in model.parameters():
            assert param.grad is not None, "every parameter should receive a gradient"

    def test_skip_connections_carry_encoder_features_forward(self):
        # zeroing every encoder weight should still let the decoder see
        # zeros (not crash / not silently skip the concat), i.e. the
        # channel-doubled DoubleConv(f*2, f) inputs really do depend on
        # both the upsampled path and the skip connection
        model = make_unet()
        for stage in model.encoder:
            for param in stage.parameters():
                param.data[...] = 0.0

        x = Tensor(np.random.randn(1, 2, 32, 32))
        out = model(x)
        assert out.data.shape == (1, 1, 32, 32)
        assert np.isfinite(out.data).all()

    def test_train_eval_toggle_propagates_to_batchnorm(self):
        model = make_unet()
        assert model.encoder[0].bn1.training is True
        model.eval()
        assert model.encoder[0].bn1.training is False

    def test_different_images_produce_different_masks(self):
        model = make_unet()
        x1 = Tensor(np.random.randn(1, 2, 32, 32))
        x2 = Tensor(np.random.randn(1, 2, 32, 32))
        out1 = model(x1)
        out2 = model(x2)
        assert not np.allclose(out1.data, out2.data)
