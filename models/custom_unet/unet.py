import torch.nn as nn
from models.unet_block import DoubleConv, Down, Up, ConvLayer, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample='bilinear'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsample = upsample

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if upsample == 'bilinear' else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, upsample)
        self.up2 = Up(512, 256 // factor, upsample)
        self.up3 = Up(256, 128 // factor, upsample)
        self.up4 = Up(128, 64, upsample)
        self.outc = OutConv(64, n_classes)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.drop(x1)
        x2 = self.down1(x1)
        x2 = self.drop(x2)
        x3 = self.down2(x2)
        x3 = self.drop(x3)
        x4 = self.down3(x3)
        x4 = self.drop(x4)
        x5 = self.down4(x4)
        x5 = self.drop(x5)

        x = self.up1(x5, x4)
        x = self.drop(x)
        x = self.up2(x, x3)
        x = self.drop(x)
        x = self.up3(x, x2)
        x = self.drop(x)
        x = self.up4(x, x1)
        x = self.drop(x)
        logits = self.outc(x)
        return logits