from __future__ import annotations

from nabla.nn.conv import Conv2D
from nabla.nn.module import Module
from nabla.tensor import Tensor


class PatchEmbedding(Module):
    """Splits an image into non-overlapping patches and linearly projects
    each one to embed_dim, producing a sequence for a Transformer to
    consume - the first step of a Vision Transformer.

    Implemented as a single Conv2D with kernel_size == stride == patch_size:
    each convolution window covers exactly one patch with no overlap, and
    the kernel's out_channels axis *is* the per-patch linear projection -
    mathematically identical to flattening each patch and running it
    through a Linear layer, just without materializing the flattened
    patches by hand. No new backward needed; Conv2D already has one.

    Args:
        img_size: Height/width of the (square) input image.
        patch_size: Height/width of each (square) patch. Must evenly
            divide img_size.
        in_channels: Number of input image channels (1 for grayscale,
            3 for RGB).
        embed_dim: Size of each patch's embedding vector.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (batch, in_channels, img_size, img_size)

        Returns:
            (batch, num_patches, embed_dim) - a sequence, ready for a
            Transformer block.
        """
        out = self.proj(x)  # (batch, embed_dim, grid, grid)
        batch, embed_dim, grid_h, grid_w = out.data.shape
        out = out.reshape((batch, embed_dim, grid_h * grid_w))  # (batch, embed_dim, num_patches)
        return out.transpose((0, 2, 1))  # (batch, num_patches, embed_dim)
