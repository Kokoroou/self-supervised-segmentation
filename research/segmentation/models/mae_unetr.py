import sys
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from .utils.image_process import random_masking, unpatchify, patchify
from .utils.pos_embed import get_2d_sincos_pos_embed

model_name = "ViTMAESegmentation"  # Name of main class


def add_model_arguments(parser):
    """
    Add model's specific arguments

    :param parser: The parser to add arguments
    """
    parser_mae = parser.add_argument_group("mae")
    parser_mae.add_argument(
        "--arch",
        "-a",
        type=str,
        help="Name of the defined autoencoder architecture to use",
    )
    parser_mae.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
        help="Masking ratio (percentage of removed patches)"
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
        "--decoder-embed-dim",
        type=int,
        help="Embedding dimension of decoder"
    )
    parser_mae.add_argument(
        "--decoder-depth",
        type=int,
        help="Depth of decoder"
    )
    parser_mae.add_argument(
        "--decoder-num-heads",
        type=int,
        help="Number of heads of decoder"
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


class UNETR_decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, grid_size):
        new_view = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in, x, hidden_states_out):
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.grid_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.grid_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.grid_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.grid_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)


class ViTMAESegmentation(nn.Module):
    """
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 encoder_embed_dim: int = 768, encoder_depth: int = 12, encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: Any = nn.LayerNorm,
                 norm_pix_loss: bool = False,
                 **kwargs):
        """
        Initialize model structure

        Args:
            img_size: Image size of input image (e.g. 224 for 224x224 image)
            patch_size: Size of each patch
            in_chans: Number of input channels (e.g. 3 for RGB)
            encoder_embed_dim: Embedding dimension of encoder
            encoder_depth: Number of encoder blocks
            encoder_num_heads: Number of heads in encoder
            decoder_embed_dim: Embedding dimension of decoder
            decoder_depth: Number of decoder blocks
            decoder_num_heads: Number of heads in decoder
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            norm_layer: Normalization layer
            norm_pix_loss: Whether to normalize pixel loss
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)

        # Summarize encoder layers for fine-tuning
        self.encoder_layers = [
            self.patch_embed,
            self.cls_token,
            self.pos_embed,
            self.blocks,
            self.norm
        ]
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # Summarize decoder layers for fine-tuning
        self.decoder_layers = [
            self.decoder_embed,
            self.mask_token,
            self.decoder_pos_embed,
            self.decoder_blocks,
            self.decoder_norm,
            self.decoder_pred
        ]
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio=0):
        """
        Forward pass through the encoder of the neural network model.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).
            mask_ratio (float): Ratio of elements to mask during random masking.

        Returns:
            torch.Tensor: Encoded output tensor.
            torch.Tensor: Mask tensor indicating the masked elements.
            torch.Tensor: Restored indices tensor for masked elements.
        """
        # Divide image into patches and embed them
        x = self.patch_embed(x)

        # Add positional embedding without classification token
        x = x + self.pos_embed[:, 1:, :]

        # Masking image patches, only keep patches that unmasked and info for restoring
        x, mask, ids_restore = random_masking(x, mask_ratio=mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = patchify(imgs, self.patch_size)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        """
        Args:
            imgs: Batch of images (shape: [N, C, H, W])
            mask_ratio: Ratio of masked patches

        Returns:
            loss: Masked autoencoder loss
            pred: Predicted patches
            mask: Mask of removed patches
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=0)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# Set recommended architectures
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def vitmae_unetr_base(checkpoint):
    model = mae_vit_base_patch16()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # Remove decoder
    for layer in model.decoder_layers:
        layer.remove()

    # Freeze encoder
    for layer in model.encoder_layers:
        layer.requires_grad = False

    # Add new decoder
    model.decoder = UNETR_decoder(
        in_channels=model.embed_dim,
        out_channels=3,
        img_size=model.img_size,
        patch_size=model.patch_size,
    )

    # Change forward function
    def forward(self, imgs):
        """
        Args:
            imgs: Batch of images (shape: [N, C, H, W])
            mask_ratio: Ratio of masked patches

        Returns:
            loss: Masked autoencoder loss
            pred: Predicted patches
            mask: Mask of removed patches
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio=0)
        pred = self.decoder(latent, imgs)

        return pred
    model.forward = types.MethodType(forward, model)

    return model
