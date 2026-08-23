from __future__ import annotations

import nabla.functional as F
from nabla.nn.batchnorm import BatchNorm2D
from nabla.nn.conv import Conv2D
from nabla.nn.container import ModuleList
from nabla.nn.module import Module
from nabla.nn.upsample import Upsample
from nabla.tensor import Tensor


class DoubleConv(Module):
    """(Conv2D -> BatchNorm2D -> ReLU) x2, the basic block a U-Net's
    encoder/decoder stages are built from - two 3x3 convolutions (with
    padding=1, so H/W are preserved) refine features at a given
    resolution before the next down/up-sampling step changes it.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        return x


class UNet(Module):
    """2D U-Net (Ronneberger et al.) for binary segmentation.

    Returns raw logits, not probabilities - apply `.sigmoid()` wherever a
    probability is actually needed (e.g. before DiceLoss, or at inference
    time). Leaving the final activation out lets the loss combine with a
    numerically stable BCEWithLogitsLoss instead of computing sigmoid
    and log separately (see docs/30-bce-with-logits.md).

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of output mask channels (1 for binary
            segmentation, one logit per pixel).
        features: Channel count at each encoder stage, shallowest first.
            The bottleneck uses `features[-1] * 2`; the decoder mirrors
            `features` in reverse.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.features = features

        self.encoder = ModuleList(
            [DoubleConv(in_channels if i == 0 else features[i - 1], f) for i, f in enumerate(features)]
        )
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # each decoder stage: Upsample + Conv2D to halve channels back down
        # to the matching skip connection's width, then a DoubleConv on
        # the concatenated (skip_channels + skip_channels = 2x) input
        reversed_features = list(reversed(features))
        up_conv_in_channels = [features[-1] * 2] + reversed_features[:-1]
        self.upsamples = ModuleList([Upsample(scale=2) for _ in features])
        self.up_convs = ModuleList(
            [
                Conv2D(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(up_conv_in_channels, reversed_features)
            ]
        )
        self.decoder = ModuleList([DoubleConv(f * 2, f) for f in reversed_features])

        self.final_conv = Conv2D(features[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # when going down the encoder we reduce spatial dimension
        # and increase conv channels -> bigger receptive field -> more abstract features
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
            x = x.max_pool2d(kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for upsample, up_conv, decoder_stage, skip in zip(
            self.upsamples, self.up_convs, self.decoder, reversed(skips)
        ):
            x = up_conv(upsample(x))
            x = F.concat([skip, x], axis=1)
            x = decoder_stage(x)

        return self.final_conv(x)
