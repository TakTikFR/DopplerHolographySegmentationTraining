import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

def match_size(src, target):
    if src.shape[-2:] != target.shape[-2:]:
        src = F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return src

def get_same_padding(kernel_size, stride, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


# --------- Conv2D + BN ----------
class Conv2DWithBN(nn.Module):
    """Custom convolution layer with batch normalization."""

    def __init__(self, filters, kernel_size, strides, padding, activation, 
                 dilation_rate=1, use_bias=False, trainable=True, **kwargs):
        super().__init__()
        self.conv = nn.LazyConv2d(
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            bias=use_bias
        )
        self.batchnorm = nn.BatchNorm2d(filters)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# --------- ConvTranspose2D + BN ----------
class Conv2DTransposeWithBN(nn.Module):
    """Custom transposed convolution layer with batch normalization."""

    def __init__(self, filters, kernel_size, strides, padding, activation, 
                 dilation_rate=1, use_bias=False, trainable=True, **kwargs):
        super().__init__()
        self.conv = nn.LazyConvTranspose2d(
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            dilation=dilation_rate,
            bias=use_bias
        )
        self.batchnorm = nn.BatchNorm2d(filters)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# --------- Conv2DOptBN ----------
class Conv2DOptBN(nn.Module):
    """Conv2D with optional BatchNorm."""

    def __init__(self, filters, kernel_size, strides, padding, activation, 
                 dilation_rate=1, use_bias=False, batch_norm=False, **kwargs):
        super().__init__()
        if padding == 'same':
            padding = get_same_padding(kernel_size, strides, dilation_rate)
        if batch_norm:
            self.conv = Conv2DWithBN(filters, kernel_size, strides, padding, activation,
                                     dilation_rate=dilation_rate, use_bias=use_bias)
        else:
            self.conv = nn.Sequential(
                nn.LazyConv2d(
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=strides,
                    padding=padding,
                    dilation=dilation_rate,
                    bias=use_bias
                ),
                nn.Identity() if activation is None else nn.ReLU(inplace=True)  # map tf activations
            )

    def forward(self, x):
        return self.conv(x)


# --------- Conv2DTransposeOptBN ----------
class Conv2DTransposeOptBN(nn.Module):
    """ConvTranspose2D with optional BatchNorm."""

    def __init__(self, filters, kernel_size, strides, padding, activation,
                 dilation_rate=1, use_bias=False, batch_norm=False, **kwargs):
        super().__init__()
        if padding == 'same':
            padding = get_same_padding(kernel_size, strides, dilation_rate)
        if batch_norm:
            self.conv = Conv2DTransposeWithBN(filters, kernel_size, strides, padding, activation,
                                              dilation_rate=dilation_rate, use_bias=use_bias)
        else:
            self.conv = nn.Sequential(
                nn.LazyConvTranspose2d(
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=strides,
                    padding=padding,
                    dilation=dilation_rate,
                    bias=use_bias
                ),
                nn.Identity() if activation is None else nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


# --------- SIN Self-Attention ----------
class SINSelfAttention(nn.Module):
    """SIN multi-head positional self-attention layer."""

    def __init__(self, ca_shape, filters, inner_filters, kernel_size, strides, padding, activation, 
                 batch_norm=False, **kwargs):
        super().__init__()
        self.ca_shape = ca_shape
        H, W = ca_shape
        self.flattened_shape = H * W

        self.conv_key = Conv2DOptBN(inner_filters, kernel_size, strides, padding, F.elu, batch_norm=batch_norm)
        self.conv_query = Conv2DOptBN(inner_filters, kernel_size, strides, padding, F.elu, batch_norm=batch_norm)
        self.conv_value = Conv2DOptBN(inner_filters, kernel_size, strides, padding, activation, batch_norm=batch_norm)

        self.conv_result = Conv2DOptBN(filters, kernel_size, strides, padding, activation, batch_norm=batch_norm)

    def forward(self, x):
        B, C, H, W = x.shape

        key = self.conv_key(x).view(B, -1, H * W)       # (B, inner_filters, N)
        query = self.conv_query(x).view(B, -1, H * W)   # (B, inner_filters, N)
        value = self.conv_value(x).view(B, -1, H * W)   # (B, inner_filters, N)

        # Attention: Q^T K
        attn = torch.bmm(query.permute(0, 2, 1), key)   # (B, N, N)

        # Mask self-connections (diag=0)
        mask = torch.eye(H * W, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        attn = attn * (1.0 - mask)

        # Scale
        attn = attn / math.sqrt(key.shape[1])

        # Softmax
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.bmm(value, attn.permute(0, 2, 1))   # (B, inner_filters, N)
        out = out.view(B, -1, H, W)

        return self.conv_result(out)


class SINBasicUNET(nn.Module):
    """Custom definition of the SIN U-Net architecture."""

    def __init__(self, base_input_shape, base_filters, batch_norm=False, activation=F.relu):
        super().__init__()
        self.base_input_shape = base_input_shape
        self.base_filters = base_filters
        self.batch_norm = batch_norm
        self.activation = activation

        # ---------------- Downscaling ----------------
        self.input_conv_1 = Conv2DOptBN(base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)
        self.down_conv_1 = Conv2DOptBN(2 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.down_conv_2 = Conv2DOptBN(4 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.down_conv_3 = Conv2DOptBN(8 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.down_conv_4 = Conv2DOptBN(16 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)

        # ---------------- Attention ----------------
        self.ca_down_2 = SINSelfAttention((base_input_shape[0] // 4, base_input_shape[1] // 4),
                                          4 * base_filters, 4 * base_filters, 1, 1, padding="same", activation=activation, batch_norm=batch_norm)
        self.ca_down_2_conv = Conv2DOptBN(4 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

        self.ca_down_3 = SINSelfAttention((base_input_shape[0] // 8, base_input_shape[1] // 8),
                                          8 * base_filters, 8 * base_filters, 1, 1, padding="same", activation=activation, batch_norm=batch_norm)
        self.ca_down_3_conv = Conv2DOptBN(8 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

        self.ca_down_4 = SINSelfAttention((base_input_shape[0] // 16, base_input_shape[1] // 16),
                                          16 * base_filters, 16 * base_filters, 1, 1, padding="same", activation=activation, batch_norm=batch_norm)
        self.ca_down_4_conv = Conv2DOptBN(16 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

        # ---------------- Straight pass ----------------
        self.straight_conv_1_1 = Conv2DOptBN(2 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm, dilation_rate=2)
        self.straight_conv_2_1 = Conv2DOptBN(4 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm, dilation_rate=2)
        self.straight_conv_3_1 = Conv2DOptBN(8 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm, dilation_rate=2)
        self.straight_conv_4_1 = Conv2DOptBN(16 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm, dilation_rate=2)

        # ---------------- Upscaling ----------------
        self.up_conv_1 = Conv2DTransposeOptBN(8 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.up_conv_ca_1 = Conv2DTransposeOptBN(8 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

        self.up_conv_2 = Conv2DTransposeOptBN(4 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.up_conv_ca_2 = Conv2DTransposeOptBN(4 * base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

        self.up_conv_3 = Conv2DTransposeOptBN(2 * base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)
        self.up_conv_4 = Conv2DTransposeOptBN(base_filters, 3, 2, padding="same", activation=activation, batch_norm=batch_norm)

        self.out_conv = Conv2DOptBN(base_filters, 3, 1, padding="same", activation=activation, batch_norm=batch_norm)

    def forward(self, x):
        # ---- Downscaling ----
        x_input = self.input_conv_1(x)
        x_down_1 = self.down_conv_1(x_input)
        x_down_2 = self.down_conv_2(x_down_1)
        x_down_3 = self.down_conv_3(x_down_2)
        x_down_4 = self.down_conv_4(x_down_3)

        # ---- Attention ----
        x_ca_2 = self.ca_down_2(x_down_2)
        x_ca_2 = x_ca_2 + x_down_2
        x_ca_2 = self.ca_down_2_conv(x_ca_2)

        x_ca_3 = self.ca_down_3(x_down_3)
        x_ca_3 = x_ca_3 + x_down_3
        x_ca_3 = self.ca_down_3_conv(x_ca_3)

        x_ca_4 = self.ca_down_4(x_down_4)
        x_ca_4 = x_ca_4 + x_down_4
        x_ca_4 = self.ca_down_4_conv(x_ca_4)

        # ---- Straight pass ----
        x_down_1 = self.straight_conv_1_1(x_down_1)
        x_ca_2 = self.straight_conv_2_1(x_down_2) + x_ca_2
        x_ca_3 = self.straight_conv_3_1(x_down_3) + x_ca_3
        x_ca_4 = self.straight_conv_4_1(x_down_4) + x_ca_4

        # ---- Upscaling ----
        
        x_up_1 = self.up_conv_1(x_ca_4)
        x_up_1 = match_size(x_up_1, x_ca_3)
        x_up_1 = x_up_1 + x_ca_3
        x_up_1 = self.up_conv_ca_1(x_up_1)

        x_up_2 = self.up_conv_2(x_up_1)
        x_up_2 = match_size(x_up_2, x_ca_2)
        x_up_2 = x_up_2 + x_ca_2
        x_up_2 = self.up_conv_ca_2(x_up_2)

        x_up_3 = self.up_conv_3(x_up_2)
        x_up_3 = match_size(x_up_3, x_down_1)
        x_up_3 = x_up_3 + x_down_1

        x_up_4 = self.up_conv_4(x_up_3)
        x_up_4 = match_size(x_up_4, x_input)
        x_up_4 = x_up_4 + x_input

        x_out = self.out_conv(x_up_4)
        return x_out


class SINVGG19(nn.Module):
    """VGG19 with multiple outputs for perceptual loss."""

    def __init__(self, base_input_shape, pretrained=True):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features

        # Extract layers at positions matching TF’s block convs
        self.block1_conv2 = nn.Sequential(*list(vgg[:4]))   # up to relu1_2
        self.block2_conv2 = nn.Sequential(*list(vgg[4:9]))  # up to relu2_2
        self.block3_conv4 = nn.Sequential(*list(vgg[9:18])) # up to relu3_4
        self.block4_conv4 = nn.Sequential(*list(vgg[18:27]))# up to relu4_4

    def forward(self, x):
        # mimic tf.keras.applications.vgg19.preprocess_input
        x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]) \
            / torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]

        out1 = self.block1_conv2(x)
        out2 = self.block2_conv2(out1)
        out3 = self.block3_conv4(out2)
        out4 = self.block4_conv4(out3)
        return out1, out2, out3, out4

class VesselSegNetwork(nn.Module):
    """Custom vessel segmentation network utilising the SIN architecture (PyTorch)."""

    def __init__(self, base_input_shape, out_channels, base_filters, batch_norm=False, trainable=True):
        """
        Args:
            base_input_shape: The input shape of the network (H, W, C).
            base_filters: Number of filters in the first layer.
            batch_norm: Whether to use batch normalisation.
            trainable: If False, parameters are frozen.
        """
        super().__init__()
        self.base_input_shape = base_input_shape
        self.base_filters = base_filters
        self.batch_norm = batch_norm

        # Define the SIN U-Net backbone
        self.common_unet = SINBasicUNET(
            base_input_shape, base_filters, batch_norm=batch_norm
        )

        # Output convolution (1x1 conv)
        self.out_conv = nn.Conv2d(
            in_channels=base_filters,  # you may need to adapt this depending on SINBasicUNET's output
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Apply trainable flag
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """Forward pass."""
        x = self.common_unet(x)
        x = self.out_conv(x)
        # x = torch.sigmoid(x)  # same as Keras activation="sigmoid"
        return x