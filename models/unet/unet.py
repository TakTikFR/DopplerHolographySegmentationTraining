import torch.nn as nn
from models.unet_block import DoubleConv, Down, Up, ConvLayer

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=64):
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
        self.outc = ConvLayer(out_channels, n_classes, kernel_size=1, act=None)

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
        return logits