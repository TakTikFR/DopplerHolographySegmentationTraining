from pathlib import Path

import torch.nn.functional as F
from torch.nn import ModuleList
import torch.nn as nn
import torch

from models.unet_block import ICNRPixelShuffleUpsample, DoubleConv, Down, Up, ConvLayer, OutConv

import re

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        upsample = 'deconv'

        self.inc = DoubleConv(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels * 2)
        self.down2 = Down(out_channels * 2, out_channels * 4)
        self.down3 = Down(out_channels * 4, out_channels * 8)
        factor = 2 if upsample == 'bilinear' else 1
        self.down4 = Down(out_channels * 8, out_channels * 16 // factor)
        self.up1 = Up(out_channels * 16, out_channels * 8 // factor, upsample)
        self.up2 = Up(out_channels * 8, out_channels * 4 // factor, upsample)
        self.up3 = Up(out_channels * 4, out_channels * 2 // factor, upsample)
        self.up4 = Up(out_channels * 2, out_channels, upsample)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x1, x, logits

class MiniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=32):
        super(MiniUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        upsample = 'deconv'

        self.inc = DoubleConv(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels*2)
        self.down2 = Down(out_channels*2, out_channels*4)
        self.down3 = Down(out_channels*4, out_channels*8)
        self.up1 = Up(out_channels*8, out_channels*4, upsample)
        self.up2 = Up(out_channels*4, out_channels*2, upsample)
        self.up3 = Up(out_channels*2, out_channels, upsample)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return x1, x, logits

class Iternet(nn.Module):
    @classmethod
    def init_from_state_dict(cls, in_channels, n_classes, weight_file):
        filename = Path(weight_file).name
        iterations = re.match(rf"IterNet_(\d+)_.*", filename).group(1)
        if iterations is not None and int(iterations) > 0:
            iterations = int(iterations)
        else:
            raise ValueError("Invalid weight file name. Expected format: 'IterNet_<iterations>_<loss>'")
        instance = cls(in_channels=in_channels, n_classes=n_classes, iterations=iterations)
        instance.load_state_dict(torch.load(weight_file))
        return instance

    def __init__(self, in_channels, n_classes, out_channels=32, iterations=3):
        super(Iternet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.iterations = iterations

        # define the network UNet2 layer
        self.model_unet = UNet(n_channels=in_channels,
                               n_classes=n_classes, out_channels=out_channels)

        # define the network MiniUNet layers
        self.model_miniunet = ModuleList(MiniUNet(
            n_channels=out_channels*2, n_classes=n_classes, out_channels=out_channels) for i in range(iterations))

    def forward(self, x):
        x1, x2, logits = self.model_unet(x)
        for i in range(self.iterations):
            x = torch.cat([x1, x2], dim=1)
            _, x2, logits = self.model_miniunet[i](x)

        return logits