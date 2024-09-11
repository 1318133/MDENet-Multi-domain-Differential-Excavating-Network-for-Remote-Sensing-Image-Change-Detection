import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Att(torch.nn.Module):
    def __init__(self, in_ch):
        super(Att, self).__init__()

        self.proj = nn.Conv2d(in_ch, in_ch*2, kernel_size=3,stride=1, padding=1, groups=2)  # generate k by conv
        self.atten = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        # xf = torch.fft.fft(x)
        xf = torch.fft.fft(torch.fft.fft(x,dim=-1),dim=-2)
        xf = xf.to(torch.float32)
        xf = self.proj(xf)
        at,xfa = xf.chunk(2, dim=1)
        att = F.gelu(at)
        xaf = att*xfa
        xf = torch.fft.ifft(torch.fft.ifft(xaf,dim=-1),dim=-2)#torch.fft.ifft(xaf)
        xf = xf.to(torch.float32)
        sxf = self.sig(xf)*x

        return sxf

