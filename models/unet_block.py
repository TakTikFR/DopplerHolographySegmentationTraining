import torch
import torch.nn as nn
import torch.nn.functional as F

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, upsample='bilinear'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        elif upsample == 'deconv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        elif upsample == 'pixelshuffle':
            self.up = ICNRPixelShuffleUpsample(in_channels, in_channels // 2, upscale_factor=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            raise ValueError("Invalid upsample method. Choose 'bilinear', 'deconv', or 'pixelshuffle'.")

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, act=nn.ReLU(inplace=True)):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = act
        
    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

def icnr_init(tensor, upscale_factor=2, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for subpixel convolution layers.
    """
    new_shape = (tensor.shape[0] // (upscale_factor ** 2),) + tensor.shape[1:]
    subkernel = torch.zeros(new_shape)
    init(subkernel)
    subkernel = subkernel.repeat(upscale_factor ** 2, 1, 1, 1)
    tensor.data.copy_(subkernel)

class ICNRPixelShuffleUpsample(nn.Module):
    """
    ICNR PixelShuffle Upsampling Layer with BatchNorm.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2, dropout=0.3):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels * (upscale_factor ** 2), kernel_size=1, padding=0, stride=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Apply ICNR initialization
        icnr_init(self.conv.conv.weight, upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    
def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)

class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query = conv1d(n_channels, n_channels//8)
        self.key   = conv1d(n_channels, n_channels//8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class UNetBlock(nn.Module):
    def __init__(self, up_channels, skip_channels, scale_factor=2, kernel_size=3, stride=1, padding=1, icnr=False, self_attention=False, final=False):
        super().__init__()
        if icnr:
            self.upsample = ICNRPixelShuffleUpsample(in_channels=up_channels, out_channels=up_channels//scale_factor, upscale_factor=scale_factor)
        else :
            self.upsample = nn.ConvTranspose2d(in_channels=up_channels, out_channels=up_channels//scale_factor, kernel_size=kernel_size, stride=scale_factor, padding=padding, output_padding=1)

        in_channels = up_channels//2 + skip_channels
        out_channels = in_channels//2 if final else in_channels
        self.bn = nn.BatchNorm2d(num_features=skip_channels)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        
        self.sa = SelfAttention(out_channels) if self_attention else None

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            print(f"x: {x.shape}, skip: {skip.shape}")
            skip = F.interpolate(skip, x.shape[-2:], mode='bilinear')
        cat_x = self.relu(torch.cat([x, self.bn(skip)], dim=1))
        x = self.conv2(self.conv1(cat_x)) 
        return self.sa(x) if self.sa else x


class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    def __init__(self, ni, nf, stride=1, ks=3, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.convpath = nn.Sequential(
                            ConvLayer(ni,  ni, ks, stride=stride, act=act),
                            ConvLayer(ni,  nf, ks, act=None),)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act=None))
        self.idpath = nn.Sequential(*idpath)

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))