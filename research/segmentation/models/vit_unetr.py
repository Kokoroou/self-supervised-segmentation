from functools import partial
from typing import Any

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from .utils.image_process import random_masking

model_name = "ViTMAESeg"  # Name of main class


def add_model_arguments(parser):
    """
    Add model's specific arguments

    :param parser: The parser to add arguments
    """
    parser_mae = parser.add_argument_group("vit_unetr")
    parser_mae.add_argument(
        "--arch",
        "-a",
        type=str,
        help="Name of the defined ViT_UNetR architecture to use",
    )
    parser_mae.add_argument(
        "--img-size",
        type=int,
        help="Size of input image"
    )
    parser_mae.add_argument(
        "--patch-size",
        type=int,
        help="Size of each patch"
    )
    parser_mae.add_argument(
        "--in-chans",
        type=int,
        help="Number of input channels"
    )
    parser_mae.add_argument(
        "--encoder-embed-dim",
        type=int,
        help="Embedding dimension of encoder"
    )
    parser_mae.add_argument(
        "--encoder-depth",
        type=int,
        help="Depth of encoder"
    )
    parser_mae.add_argument(
        "--encoder-num-heads",
        type=int,
        help="Number of heads of encoder"
    )
    parser_mae.add_argument(
        "--mlp-ratio",
        type=float,
        help="Ratio of mlp hidden dimension to embedding dimension"
    )
    parser_mae.add_argument(
        "--norm-pix-loss",
        action="store_true",
        help="Normalize pixel loss"
    )


class UNetRConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class UNetRDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class ViTMAESeg(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 encoder_embed_dim: int = 768, encoder_depth: int = 12, encoder_num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 norm_layer: Any = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 **kwargs):
        super(ViTMAESeg, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder_embed_dim = encoder_embed_dim

        # --------------------------------------------------------------------------
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)

        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim),
        #                               requires_grad=True)

        # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, encoder_embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)

        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder

        # Decoder 1
        self.d1 = UNetRDeconvBlock(encoder_embed_dim, 512)
        self.s1 = nn.Sequential(
            UNetRDeconvBlock(encoder_embed_dim, 512),
            UNetRConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            UNetRConvBlock(512 + 512, 512),
            UNetRConvBlock(512, 512)
        )

        # Decoder 2
        self.d2 = UNetRDeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            UNetRDeconvBlock(encoder_embed_dim, 256),
            UNetRConvBlock(256, 256),
            UNetRDeconvBlock(256, 256),
            UNetRConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            UNetRConvBlock(256 + 256, 256),
            UNetRConvBlock(256, 256)
        )

        # Decoder 3
        self.d3 = UNetRDeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            UNetRDeconvBlock(encoder_embed_dim, 128),
            UNetRConvBlock(128, 128),
            UNetRDeconvBlock(128, 128),
            UNetRConvBlock(128, 128),
            UNetRDeconvBlock(128, 128),
            UNetRConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            UNetRConvBlock(128 + 128, 128),
            UNetRConvBlock(128, 128)
        )

        # Decoder 4
        self.d4 = UNetRDeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            UNetRConvBlock(3, 64),
            UNetRConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            UNetRConvBlock(64 + 64, 64),
            UNetRConvBlock(64, 64)
        )

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def load_custom_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # if isinstance(param, Parameter):
            #     # backwards compatibility for serialized parameters
            #     param = param.data
            own_state[name].copy_(param)

    def forward(self, inputs):
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        """ Patch + Position Embeddings """
        # Divide image into patches and embed them
        x = self.patch_embed(inputs)

        # Add positional embedding without classification token
        x = x + self.pos_embed[:, 1:, :]

        # Masking image patches, only keep patches that unmasked and info for restoring
        x, mask, ids_restore = random_masking(x, mask_ratio=0)

        # # Append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

        # x = self.norm(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        # Reshaping
        batch = inputs.shape[0]
        z0 = inputs.view((batch, self.in_chans, self.img_size, self.img_size))

        shape = (batch,
                 self.encoder_embed_dim,
                 self.img_size // self.patch_size,
                 self.img_size // self.patch_size)

        z3 = z3.reshape(shape)
        z6 = z6.reshape(shape)
        z9 = z9.reshape(shape)
        z12 = z12.reshape(shape)

        # Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        """ Output """
        output = self.output(x)

        return output


def vit_mae_seg_base(**kwargs):
    model = ViTMAESeg(patch_size=16, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                      decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                      mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_mae_seg_large(**kwargs):
    model = ViTMAESeg(patch_size=16, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
                      decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                      mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_mae_seg_huge(**kwargs):
    model = ViTMAESeg(patch_size=14, encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
                      decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                      mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
