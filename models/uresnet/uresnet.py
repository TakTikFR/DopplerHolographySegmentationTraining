import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin
from fastcore.all import *
from fastai.vision import *
from fastai.vision.all import *
from models.unet_block import UNetBlock, ICNRPixelShuffleUpsample, ConvLayer, ResBlock
import model_utils

def _get_first_layer(m):
    "Access first layer of a model"
    c,p,n = m,None,None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p,c=c,getattr(c,n)
    return c,p,n

def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    if n_in==1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
    elif n_in==2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:,:2] * 1.5
    else:
        # keep 3 channels weights and set others to null
        new_layer.weight.data[:,:3] = previous_layer.weight.data
        new_layer.weight.data[:,3:].zero_()

def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)

class UResNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_in=1, num_classes=2, pretrained=True, dropout=0.2, pixel_shuffle=False, self_attention=False):
        super().__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        model_utils._update_first_layer_input(resnet, n_in, pretrained)

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

        # Decoder with skip connections, batch norm, and dropout
        self.decoder4 = UNetBlock(512, 256, icnr=pixel_shuffle)
        self.decoder3 = UNetBlock(512, 128, icnr=pixel_shuffle, self_attention=self_attention)
        self.decoder2 = UNetBlock(384, 64, icnr=pixel_shuffle)
        self.decoder1 = UNetBlock(256, 64, final=True, icnr=pixel_shuffle)

        self.shuf = ICNRPixelShuffleUpsample(96, 96)

        self.res = ResBlock(96 + n_in, 96 + n_in)

        # Final output layer
        self.final_conv = nn.Conv2d(96 + n_in, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        w, h = x.shape[-2:]
        enc0 = self.l0(x) # 3 x256x256 -> 64 x256x256

        enc1 =self.l1(self.ds1(enc0))  # 64 x128x128
        
        enc2 = self.l2(self.ds2(enc1))  # 128 x64x64
        
        enc3 = self.l3(self.ds3(enc2))  # 256 x32x32
        
        enc4 = self.l4(self.ds4(enc3))  # 512 x16x16

        # Bottleneck
        mid = self.bottleneck(enc4) # 1024 x16x16

        # Decoder with skip connections
        dec4 = self.decoder4(mid, enc3)  # 512
        
        dec3 = self.decoder3(dec4, enc2) # 256
        
        dec2 = self.decoder2(dec3, enc1) # 128
        
        dec1 = self.decoder1(dec2, enc0)  # 64
        
        if dec1.shape[-1] * dec1.shape[-2] < w * h:
            dec1 = self.shuf(dec1)

        x = torch.cat([dec1,x], dim=1)

        # Final segmentation output
        x = self.final_conv(x) # 2

        if x.shape[-2:] != (w, h):
            x = F.interpolate(x, size=(w, h), mode="bilinear")
            
        return x
