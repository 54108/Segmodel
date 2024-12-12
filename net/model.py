import torch
import torch.nn as nn

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(self.dw(self.bn1(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class base_net(nn.Module):
    def __init__(self, num_class = 4, channel = 3, use_bottleneck = False):
        super().__init__()
        self.num_class = num_class
        self.channel = channel

        """Encoder"""
        self.conv_in = nn.Conv2d(self.channel, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        """Bottle Neck"""
        if use_bottleneck:
            # self.b1 = BottleNeckBlock(16)
            # self.b2 = BottleNeckBlock(32)
            # self.b3 = BottleNeckBlock(64)
            self.b1 = BottleNeckBlock(128)

        # """Decoder"""
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, self.num_class, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        # skip1 = self.b1(skip1) if hasattr(self, 'b1') else skip1
        # skip2 = self.b2(skip2) if hasattr(self, 'b2') else skip2
        # skip3 = self.b3(skip3) if hasattr(self, 'b3') else skip3

        """BottleNeck"""
        if hasattr(self, 'b1'):
            x = self.b1(x)

        # """Decoder"""
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
    
class self_net(nn.Module):
    def __init__(self, num_class = 4, channel = 3):
        super().__init__()
        self.num_class = num_class
        self.channel = channel

        self.class0 = base_net(num_class = 4, channel = channel, use_bottleneck = True)

    def forward(self, x):
        x = self.class0(x)
        return x