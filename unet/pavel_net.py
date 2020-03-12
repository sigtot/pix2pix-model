import torch
from torch import nn
import torch.nn.functional as F


class EncodeModule(nn.Module):
    def __init__(self, in_c, out_c, batchnorm=False):
        super(EncodeModule, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv', nn.Conv2d(in_c, out_c, 4, stride=2, padding=1))
        if batchnorm:
            self.layers.add_module('bn', nn.BatchNorm2d(out_c))
        self.layers.add_module('relu', nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out = self.layers(x)
        #print(out.size())
        return out


class DecodeModule(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super(DecodeModule, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.layers = nn.Sequential()
        self.layers.add_module('bn', nn.BatchNorm2d(out_c*2))
        if dropout:
            self.layers.add_module('do', nn.Dropout2d(p=0.5, inplace=True))
        self.layers.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dw = x2.size(2) - x1.size(2)
        dh = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        #print("x1", x1.size())
        #print("x2", x2.size())
        x = torch.cat([x1, x2], dim=1)
        out = self.layers(x)
        #print(out.size())
        return out


class PavelNet(nn.Module):
    def __init__(self):
        super(PavelNet, self).__init__()
        # C64-C128-C256-C512-C512-C512-C512-C512
        self.e1 = EncodeModule(3, 64, batchnorm=True)  # 256 -> 128
        self.e2 = EncodeModule(64, 128)  # 128 -> 64
        self.e3 = EncodeModule(128, 256)  # 64 -> 32
        self.e4 = EncodeModule(256, 512)  # 32 -> 16
        self.e5 = EncodeModule(512, 512)  # 16 -> 8
        self.e6 = EncodeModule(512, 512)  # 8 -> 4
        self.e7 = EncodeModule(512, 512)  # 4 -> 2
        self.e8 = EncodeModule(512, 512)  # 2 -> 1

        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.d1 = DecodeModule(512, 512, dropout=True)
        self.d2 = DecodeModule(1024, 512, dropout=True)
        self.d3 = DecodeModule(1024, 512, dropout=True)
        self.d4 = DecodeModule(1024, 512)
        self.d5 = DecodeModule(1024, 256)
        self.d6 = DecodeModule(512, 128)
        self.d7 = DecodeModule(256, 64)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        x8 = self.e8(x7)

        y = self.d1(x8, x7)
        y = self.d2(y, x6)
        y = self.d3(y, x5)
        y = self.d4(y, x4)
        y = self.d5(y, x3)
        y = self.d6(y, x2)
        y = self.d7(y, x1)
        return self.out(y)
