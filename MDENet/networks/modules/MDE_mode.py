import math
import torch
from torch import nn

from networks.modules.ATT import SSFC,Att
from networks.modules.CMConv import CMConv


class MSDConv_SSFC(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        # native feature maps
        self.native = nn.Sequential(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        self.aux = nn.Sequential(
            CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
                   bias=False),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(inplace=True),
        )

        self.att = SSFC(aux_ch)

    def forward(self, x):
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]

class Conv_our_feature_output(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(Conv_our, self).__init__()
        self.out_ch = out_ch
        #native_ch = math.ceil(out_ch / ratio)
        #aux_ch = native_ch * (ratio - 1)
        med_ch = math.ceil(out_ch / 2)
        #med_ch = out_ch /2

        # native feature maps
        self.native1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native3 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native5 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 1, stride, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        # self.aux = nn.Sequential(
        #     CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
        #            bias=False),
        #     nn.BatchNorm2d(aux_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.att = Att(med_ch)

    def forward(self, x):
        x1 = self.native1(x)
        x3 = self.native3(x)
        x5 = self.native5(x)
        xa = self.att(x1+x3+x5)
        xn = self.c1(x)
        out = torch.cat([xa, xn], dim=1)
        return out[:, :self.out_ch, :, :],xa,xn

class Conv_our(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(Conv_our, self).__init__()
        self.out_ch = out_ch
        #native_ch = math.ceil(out_ch / ratio)
        #aux_ch = native_ch * (ratio - 1)
        med_ch = math.ceil(out_ch / 2)
        #med_ch = out_ch /2

        # native feature maps
        self.native1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native3 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native5 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 1, stride, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        # self.aux = nn.Sequential(
        #     CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
        #            bias=False),
        #     nn.BatchNorm2d(aux_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.att = Att(med_ch)

    def forward(self, x):
        x1 = self.native1(x)
        x3 = self.native3(x)
        x5 = self.native5(x)
        xa = self.att(x1+x3+x5)
        xn = self.c1(x)
        out = torch.cat([xa, xn], dim=1)
        return out[:, :self.out_ch, :, :]


class Conv_our_d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(Conv_our_d, self).__init__()
        self.out_ch = out_ch
        #native_ch = math.ceil(out_ch / ratio)
        #aux_ch = native_ch * (ratio - 1)
        med_ch = math.ceil(out_ch / 2)
        #med_ch = out_ch /2

        # native feature maps
        self.native1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native3 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.native5 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 3, stride, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 1, stride, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        # self.aux = nn.Sequential(
        #     CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
        #            bias=False),
        #     nn.BatchNorm2d(aux_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.att = Att(med_ch)

    def forward(self, x):
        x1 = self.native1(x)
        x3 = self.native3(x)
        x5 = self.native5(x)
        xa = x1+x3+x5
        xn = self.c1(x)
        out = torch.cat([xa, xn], dim=1)
        return out[:, :self.out_ch, :, :]


class Conv_our_c(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(Conv_our_c, self).__init__()
        self.out_ch = out_ch
        #native_ch = math.ceil(out_ch / ratio)
        #aux_ch = native_ch * (ratio - 1)
        med_ch = math.ceil(out_ch / 2)
        #med_ch = out_ch /2

        # # native feature maps
        # self.native1 = nn.Sequential(
        #     nn.Conv2d(in_ch, med_ch, 3, stride, padding=1, dilation=1, bias=False),
        #     nn.BatchNorm2d(med_ch),
        #     nn.ReLU(inplace=True),
        # )

        # self.native3 = nn.Sequential(
        #     nn.Conv2d(in_ch, med_ch, 3, stride, padding=3, dilation=3, bias=False),
        #     nn.BatchNorm2d(med_ch),
        #     nn.ReLU(inplace=True),
        # )

        # self.native5 = nn.Sequential(
        #     nn.Conv2d(in_ch, med_ch, 3, stride, padding=5, dilation=5, bias=False),
        #     nn.BatchNorm2d(med_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 1, stride, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        self.c12 = nn.Sequential(
            nn.Conv2d(in_ch, med_ch, 1, stride, bias=False),
            nn.BatchNorm2d(med_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        # self.aux = nn.Sequential(
        #     CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
        #            bias=False),
        #     nn.BatchNorm2d(aux_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.att = Att(med_ch)

    def forward(self, x):
        # x1 = self.native1(x)
        # x3 = self.native3(x)
        # x5 = self.native5(x)
        # xa = x1+x3+x5
        xn = self.c1(x)
        xnn = self.c1(x)
        out = torch.cat([xnn, xn], dim=1)
        return out[:, :self.out_ch, :, :]


class Conv_encoder_sp(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(Conv_encoder_sp, self).__init__()
        self.out_ch = out_ch
        #native_ch = math.ceil(out_ch / ratio)
        #aux_ch = native_ch * (ratio - 1)
        med_ch = math.ceil(out_ch / 2)
        #med_ch = out_ch /2

        # native feature maps
        self.native1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.native3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.native5 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch*3, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(in_ch*2, out_ch, 1, stride, bias=False),
        #     nn.BatchNorm2d(med_ch),
        #     nn.ReLU(inplace=True),
        # )

        # auxiliary feature maps
        # self.aux = nn.Sequential(
        #     CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=int(native_ch / 4), dilation=dilation,
        #            bias=False),
        #     nn.BatchNorm2d(aux_ch),
        #     nn.ReLU(inplace=True),
        # )

        # self.att = Att_ffi(in_ch)

    def forward(self, x):
        x1 = self.native1(x)
        x3 = self.native3(x)
        x5 = self.native5(x)
        xc = torch.cat((x1,x3,x5),dim=1)
        out = self.c1(xc)
        # xa = self.att(x)
        # xcc = torch.cat((xc,xa),dim=1)
        # out = self.c2(xcc)
        # out = xa+xn
        return out[:, :self.out_ch, :, :]
