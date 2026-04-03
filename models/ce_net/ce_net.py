from functools import reduce
import torch
import torch.nn as nn
from fastai.vision import models
from models.unet_block import UNetBlock, ResBlock
import torch.nn.functional as F
import model_utils
import numpy as np

class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if(shape_x[-1] < self.output_size[-1]):
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1]))
            paddzero = paddzero.to('cuda:0')
            x = torch.cat((x, paddzero), axis=-1)

        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x

class RMPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[2, 3, 5, 6], dropout=0.3):
        super(RMPBlock, self).__init__()
        self.paths = nn.ModuleList()
        for size in pool_sizes:
            self.paths.append(nn.Sequential(
                AdaptiveAvgPool2dCustom(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ))
        self.conv = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        path_outputs = [x] + [F.upsample(path(x), size=(h,w), mode='bilinear') for path in self.paths]
        x = torch.cat(path_outputs, dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x

class DacBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5], dropout=0.3):
        super(DacBlock, self).__init__()
        self.branches = nn.ModuleList()

        b1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1))
        b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1))
        b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1))
        
        self.branches.extend([b1, b2, b3, b4])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_outputs = [x] + [branch(x) for branch in self.branches]
        x = reduce(lambda a, b: a + b, branch_outputs)
        x = self.relu(x)
        return x

# %% ../../nbs/21_vision.learner.ipynb 14
def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = model_utils._get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        model_utils._load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)

class CE_Net(nn.Module):
    def __init__(self, n_in=1, num_classes=2, pretrained=True, dropout=0.2, pixel_shuffle=False, self_attention=False):
        super().__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        _update_first_layer(resnet, n_in, pretrained)

        # Extract ResNet34 encoder layers
        self.l0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels

        self.ds1 = resnet.maxpool
        self.l1 = resnet.layer1  # 64

        self.ds2 = resnet.layer2[0]
        self.l2 = resnet.layer2[1:]  # 128

        self.ds3 = resnet.layer3[0]
        self.l3 = resnet.layer3[1:]  # 256

        self.ds4 = resnet.layer4[0]
        self.l4 = resnet.layer4[1:]  # 512

        self.dac = DacBlock(in_channels=512, out_channels=512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=dropout)
        )  # 1024

        self.rmp = RMPBlock(in_channels=512, out_channels=512)

        # Decoder with skip connections, batch norm, and dropout
        self.decoder4 = UNetBlock(512, 256, icnr=pixel_shuffle)
        self.decoder3 = UNetBlock(512, 128, icnr=pixel_shuffle, self_attention=self_attention)
        self.decoder2 = UNetBlock(384, 64, icnr=pixel_shuffle)
        self.decoder1 = UNetBlock(256, 64, final=True, icnr=pixel_shuffle)

        # self.shuf = ICNRPixelShuffleUpsample(96, 96)

        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        w, h = x.shape[-2:]
        enc0 = self.l0(x) # 3 x256x256 -> 64 x256x256

        enc1 =self.l1(self.ds1(enc0))  # 64 x128x128
        
        enc2 = self.l2(self.ds2(enc1))  # 128 x64x64
        
        enc3 = self.l3(self.ds3(enc2))  # 256 x32x32
        
        enc4 = self.l4(self.ds4(enc3))  # 512 x16x16
        
        enc4 = self.dac(enc4)  # 512 x16x16

        # Bottleneck
        mid = self.bottleneck(enc4) # 1024 x16x16

        mid = self.rmp(mid)  # 512 x16x16

        # Decoder with skip connections
        dec4 = self.decoder4(mid, enc3)  # 512
        
        dec3 = self.decoder3(dec4, enc2) # 256
        
        dec2 = self.decoder2(dec3, enc1) # 128
        
        dec1 = self.decoder1(dec2, enc0)  # 64

        # Final segmentation output
        x = self.final_conv(dec1) # 2

        if x.shape[-2:] != (w, h):
            x = F.interpolate(x, size=(w, h), mode="bilinear")
            
        return x
        
        
        