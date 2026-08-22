from __future__ import annotations

import numpy as np

import nabla.functional as F
from nabla.backend import get_array_module
from nabla.nn.container import ModuleList
from nabla.nn.layernorm import LayerNorm
from nabla.nn.linear import Linear
from nabla.nn.module import Module
from nabla.nn.patch_embedding import PatchEmbedding
from nabla.nn.transformer import TransformerBlock
from nabla.tensor import Tensor


class VisionTransformer(Module):
    """Image classification via the ViT architecture (Dosovitskiy et al.,
    "An Image is Worth 16x16 Words"): split the image into patches, treat
    them as a sequence, and run the exact same Transformer stack used for
    text - the only vision-specific part is turning pixels into a sequence
    in the first place.

    Architecture: PatchEmbedding -> prepend a learnable CLS token -> add a
    learnable positional embedding -> TransformerBlock stack (no mask - a
    full image needs no causal restriction, every patch may attend to
    every other patch) -> LayerNorm -> classification head applied to the
    CLS token's final representation only (picked out via `x[:, 0]`).

    Args:
        img_size: Height/width of the (square) input image.
        patch_size: Height/width of each (square) patch.
        in_channels: Number of input image channels.
        num_classes: Number of output classes.
        embed_dim: Size of the token/patch embedding.
        num_heads: Number of attention heads per TransformerBlock.
        hidden_dim: Hidden size of each TransformerBlock's feedforward sublayer.
        num_layers: Number of stacked TransformerBlocks.
        dropout: Dropout probability used throughout.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Tensor((np.random.randn(1, 1, embed_dim) * 0.02).astype(np.float32), requires_grad=True)
        self.pos_embedding = Tensor(
            (np.random.randn(1, num_patches + 1, embed_dim) * 0.02).astype(np.float32), requires_grad=True
        )

        self.blocks = ModuleList(
            [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm_out = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Args:
            images: (batch, in_channels, img_size, img_size)

        Returns:
            (batch, num_classes) class logits.
        """
        batch = images.data.shape[0]
        x = self.patch_embed(images)  # (batch, num_patches, embed_dim)

        xp = get_array_module(self.cls_token.data)
        cls_tokens = self.cls_token + Tensor(xp.zeros((batch, 1, self.cls_token.data.shape[-1])))
        x = F.concat([cls_tokens, x], axis=1)  # (batch, num_patches + 1, embed_dim)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)  # no mask - every patch may attend to every other patch
        x = self.norm_out(x)

        cls_out = x[:, 0]  # (batch, embed_dim) - the CLS token's final representation
        return self.head(cls_out)
