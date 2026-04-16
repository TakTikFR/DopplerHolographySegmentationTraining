import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin
from fastcore.all import *
from fastai.vision import *
from fastai.vision.all import *
from models.unet_block import UNetBlock, ICNRPixelShuffleUpsample, ConvLayer, ResBlock
import model_utils

class UResNet(nn.Module, PyTorchModelHubMixin):
    @classmethod
    def init_from_state_dict(cls, in_channels, n_classes, weight_file):
        filename = Path(weight_file).name
        pattern = re.compile(r"^UResNet(?:_(attention))?_([A-Za-z]+).+$")

        m = pattern.match(filename)
        if m:
            has_attention = m.group(1) is not None
            upsample_method = m.group(2)
            print(filename, has_attention, upsample_method)

            if upsample_method is None or upsample_method not in ['bilinear', 'deconv', 'pixelshuffle']:
                raise ValueError("Invalid upsample method in weight file name. Expected format: 'UResNet_(attention)?_<upsample>_<loss>'. Upsample should be 'bilinear', 'deconv', or 'pixelshuffle'. Attention is optional and indicated by 'attention_' prefix.")
    
        else:
            raise ValueError(f"Invalid weight file name : {filename}. Expected format: 'UResNet_(attention)?_<upsample>_<loss>'. Upsample should be 'bilinear', 'deconv', or 'pixelshuffle'. Attention is optional and indicated by 'attention_' prefix.")
        
        instance = cls(in_channels=in_channels, n_classes=n_classes, upsample_method=upsample_method, self_attention=has_attention)
        instance.load_state_dict(torch.load(weight_file))
        return instance

    def __init__(self, in_channels=1, n_classes=2, pretrained=True, dropout=0.2, upsample_method='bilinear', self_attention=False):
        super().__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        model_utils._update_first_layer_input(resnet, in_channels, pretrained)

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
        self.decoder4 = UNetBlock(512, 256, upsample_method=upsample_method)
        self.decoder3 = UNetBlock(512, 128, upsample_method=upsample_method, self_attention=self_attention)
        self.decoder2 = UNetBlock(384, 64, upsample_method=upsample_method)
        self.decoder1 = UNetBlock(256, 64, final=True, upsample_method=upsample_method)

        self.shuf = ICNRPixelShuffleUpsample(96, 96)

        self.res = ResBlock(96 + in_channels, 96 + in_channels)

        # Final output layer
        self.final_conv = nn.Conv2d(96 + in_channels, n_classes, kernel_size=1)

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
