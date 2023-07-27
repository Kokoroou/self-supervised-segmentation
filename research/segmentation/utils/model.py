from typing import Any

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from transformers import ViTMAEConfig, PreTrainedModel


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
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def load_custom_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    def forward(self, inputs):
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        """ Patch + Position Embeddings """
        # Divide image into patches and embed them
        x = self.patch_embed(inputs)

        # Add positional embedding without classification token
        x = x + self.pos_embed[:, 1:, :]

        # Apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

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


class ViTMAESegModel(PreTrainedModel):
    config_class = ViTMAEConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model = ViTMAESeg(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            encoder_embed_dim=config.hidden_size,
            encoder_depth=config.num_hidden_layers,
            encoder_num_heads=config.num_attention_heads,
        )

    def forward(self, pixel_values):
        return self.model(pixel_values)
