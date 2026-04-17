from __future__ import annotations
import itertools
from collections.abc import Sequence
from pathlib import Path
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg
from .SwinViT.SwinViT import SwinTransformer, SwinTransformer_1, SwinTransformer_2, SwinTransformer_3
from .encoders import Encoder1, Encoder2, Encoder3, Encoder4
from .decoders import Decoder1, Decoder2, Decoder3, Decoder4, DecoderF


import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

import re

rearrange, _ = optional_import("einops", name="rearrange")

class PatchMergingV2(nn.Module):

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}




__all__ = [
    "SimpleUNET",
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


########################################### remove the concatenate and having simple decoder:
class CSTANet(nn.Module):

    patch_size: Final[int] = 2

    @classmethod
    def init_from_state_dict(cls, in_channels, n_classes, weight_file):
        filename = Path(weight_file).name
        img_size = re.match(rf"CSTANet_(\d+)_(\d)_.*", filename).group(1)
        spatial_dim = re.match(rf"CSTANet_(\d+)_(\d)_.*", filename).group(2)
        if img_size is not None and int(img_size) > 0 and spatial_dim is not None and int(spatial_dim) in (2, 3):
            img_size = int(img_size)
            spatial_dim = int(spatial_dim)
        else:
            raise ValueError("Invalid weight file name. Expected format: 'CSTANet_<img_size>_<spatial_dim>_<loss>'")
        instance = cls(in_channels=in_channels, n_classes=n_classes, img_size=img_size, spatial_dims=spatial_dim)
        instance.load_state_dict(torch.load(weight_file))
        return instance

    def __init__(
        self,   
        img_size: Sequence[int] | int,
        in_channels: int,
        n_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        # print(f"{feature_size=}")

        self.swinViT_1 = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        self.swinViT_2 = SwinTransformer_1(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        self.swinViT_3 = SwinTransformer_3(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        self.swinViT_4 = SwinTransformer_2(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        ## decoders
        self.decoder1 = Decoder1(spatial_dims=spatial_dims, feature_size=feature_size)
        self.decoder2 = Decoder2(spatial_dims=spatial_dims, feature_size=feature_size)
        self.decoder3 = Decoder3(spatial_dims=spatial_dims, feature_size=feature_size)
        self.decoder4 = Decoder4(spatial_dims=spatial_dims, feature_size=feature_size)
        self.decoderF = DecoderF(spatial_dims=spatial_dims, feature_size=feature_size)

        ## enocders
        self.encoder1 = Encoder1(spatial_dims=spatial_dims, feature_size=feature_size)
        self.encoder2 = Encoder2(spatial_dims=spatial_dims, feature_size=feature_size)
        self.encoder3 = Encoder3(spatial_dims=spatial_dims, feature_size=feature_size)
        self.encoder4 = Encoder4(spatial_dims=spatial_dims, feature_size=feature_size)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels= 2*feature_size, out_channels=n_classes)


    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):

        x_in_conv =self.encoder0(x_in)

        ## first Unet
        hidden_states_out_1 = self.swinViT_1(x_in, self.normalize)

        # print(f"{len(hidden_states_out_1)=}")
        # print(f"{hidden_states_out_1[0].shape=}")

        hidden_states_out_1 = self.encoder1(hidden_states_out_1)

        branch_1 = self.decoder1(hidden_states_out_1)

        ## Compute sizes for 2D input
        ln_1 = x_in.shape[2]
        ln_2 = ln_1 // 2
        ln_3 = ln_1 // 4

        # UNet 2
        L,H = ln_1//4, 3*ln_1//4
        hidden_states_out_2 = self.swinViT_2(x_in[:, :, L:H, L:H], self.normalize)
        hidden_states_out_2 = self.encoder2(hidden_states_out_2)
        branch_2 = self.decoder2(hidden_states_out_2)

        # UNet 3
        L,H = ln_2//4, 3*ln_2//4
        hidden_states_out_3 = self.swinViT_3(x_in[:, :, L:H, L:H], self.normalize)
        hidden_states_out_3 = self.encoder3(hidden_states_out_3)
        branch_3 = self.decoder3(hidden_states_out_3)

        # UNet 4
        L,H = ln_3//4, 3*ln_3//4
        hidden_states_out_4 = self.swinViT_4(x_in[:, :, L:H, L:H], self.normalize)
        hidden_states_out_4 = self.encoder4(hidden_states_out_4)
        branch_4 = self.decoder4(hidden_states_out_4)

        # Final decoder
        cat_out = [x_in_conv, branch_1, branch_2, branch_3, branch_4]
        output = self.decoderF(cat_out)

        # Last layer
        output = self.out(output)

        return output

