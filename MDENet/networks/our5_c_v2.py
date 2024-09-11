import torch.nn as nn
import torch
from thop import profile
from torchsummary import summary
import torch.nn.functional as F
from torch import einsum
from networks.modules.MDE_mode import MSDConv_SSFC,Conv_our
from einops import rearrange, reduce

class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class FeedForward(nn.Module):
    def __init__(self, inchannel, outchannel, bias):
        super(FeedForward, self).__init__()

        # hidden_features = int(dim*ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, groups=inchannel, bias=bias)

        self.project_out = nn.Conv2d(outchannel, outchannel, kernel_size=1, bias=bias)

    def forward(self, x):
        # x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            Conv_our(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Conv_our(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class MDENet(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super(MDENet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv2_1 = First_DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv2_2 = First_DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv3_1 = First_DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv3_2 = First_DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv4_1 = First_DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv4_2 = First_DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv5_1 = First_DoubleConv(int(512 * ratio), int(1024 * ratio))
        self.Conv5_2 = First_DoubleConv(int(512 * ratio), int(1024 * ratio))

        self.Conv11_1 = First_DoubleConv(int(64 * ratio), int(64 * ratio))
        self.Conv11_2 = First_DoubleConv(int(64 * ratio), int(64 * ratio))
        self.Conv12_1 = First_DoubleConv(int(128 * ratio), int(128 * ratio))
        self.Conv12_2 = First_DoubleConv(int(128 * ratio), int(128 * ratio))
        self.Conv13_1 = First_DoubleConv(int(256 * ratio), int(256 * ratio))
        self.Conv13_2 = First_DoubleConv(int(256 * ratio), int(256 * ratio))
        self.Conv14_1 = First_DoubleConv(int(512 * ratio), int(512 * ratio))
        self.Conv14_2 = First_DoubleConv(int(512 * ratio), int(512 * ratio))
        self.Conv15_1 = First_DoubleConv(int(1024 * ratio), int(1024 * ratio))
        self.Conv15_2 = First_DoubleConv(int(1024 * ratio), int(1024 * ratio))

        self.Conv21_1 = DoubleConv(int(64 * ratio), int(64 * ratio))
        self.Conv21_2 = DoubleConv(int(64 * ratio), int(64 * ratio))
        self.Conv22_1 = DoubleConv(int(128 * ratio), int(128 * ratio))
        self.Conv22_2 = DoubleConv(int(128 * ratio), int(128 * ratio))
        self.Conv23_1 = DoubleConv(int(256 * ratio), int(256 * ratio))
        self.Conv23_2 = DoubleConv(int(256 * ratio), int(256 * ratio))
        self.Conv24_1 = DoubleConv(int(512 * ratio), int(512 * ratio))
        self.Conv24_2 = DoubleConv(int(512 * ratio), int(512 * ratio))
        self.Conv25_1 = DoubleConv(int(1024 * ratio), int(1024 * ratio))
        self.Conv25_2 = DoubleConv(int(1024 * ratio), int(1024 * ratio))

        # self.Conv31_1 = First_DoubleConv(in_ch, int(64 * ratio))
        # self.Conv31_2 = First_DoubleConv(in_ch, int(64 * ratio))
        # self.Conv32_1 = First_DoubleConv(int(64 * ratio), int(128 * ratio))
        # self.Conv32_2 = First_DoubleConv(int(64 * ratio), int(128 * ratio))
        # self.Conv33_1 = First_DoubleConv(int(128 * ratio), int(256 * ratio))
        # self.Conv33_2 = First_DoubleConv(int(128 * ratio), int(256 * ratio))
        # self.Conv34_1 = First_DoubleConv(int(256 * ratio), int(512 * ratio))
        # self.Conv34_2 = First_DoubleConv(int(256 * ratio), int(512 * ratio))
        # self.Conv35_1 = First_DoubleConv(int(512 * ratio), int(1024 * ratio))
        # self.Conv35_2 = First_DoubleConv(int(512 * ratio), int(1024 * ratio))

        self.Up5 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up52 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up50 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up_conv5 = First_DoubleConv(int(1024 * ratio), int(512 * ratio))
        # self.Conv_5_1x1 = nn.Conv2d(int(512 * ratio), out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv5_1x1 = nn.Conv2d(int(1536 * ratio), int(512 * ratio), kernel_size=1, stride=1, padding=0)

        self.Up4 = nn.ConvTranspose2d(int(512 * ratio), int(256 * ratio), 2, stride=2)
        # self.Up42 = nn.ConvTranspose2d(int(512 * ratio), int(256 * ratio), 2, stride=2)
        self.Up_conv4 = First_DoubleConv(int(512 * ratio), int(256 * ratio))
        # self.Conv_4_1x1 = nn.Conv2d(int(256 * ratio), out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv4_1x1 = nn.Conv2d(int(1536 * ratio), int(512 * ratio), kernel_size=1, stride=1, padding=0)

        self.Up3 = nn.ConvTranspose2d(int(256 * ratio), int(128 * ratio), 2, stride=2)
        # self.Up32 = nn.ConvTranspose2d(int(256 * ratio), int(128 * ratio), 2, stride=2)
        self.Up_conv3 = First_DoubleConv(int(256 * ratio), int(128 * ratio))
        # self.Conv_3_1x1 = nn.Conv2d(int(128 * ratio), out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv3_1x1 = nn.Conv2d(int(768 * ratio), int(256 * ratio), kernel_size=1, stride=1, padding=0)

        self.Up2 = nn.ConvTranspose2d(int(128 * ratio), int(64 * ratio), 2, stride=2)
        # self.Up22 = nn.ConvTranspose2d(int(128 * ratio), int(64 * ratio), 2, stride=2)
        self.Up_conv2 = First_DoubleConv(int(128 * ratio), int(64 * ratio))
        self.Conv2_1x1 = nn.Conv2d(int(384 * ratio), int(128 * ratio), kernel_size=1, stride=1, padding=0)
        self.Conv1_1x1 = nn.Conv2d(int(192 * ratio), int(64 * ratio), kernel_size=1, stride=1, padding=0)

        self.Conv_1x1 = nn.Conv2d(int(64 * ratio), out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # encoding
        # x1, x2 = torch.unsqueeze(x1[0], dim=0), torch.unsqueeze(x1[1], dim=0)
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)
        z1 = torch.abs(torch.sub(c1_1, c1_2))
        c1_1 = self.Conv11_1(c1_1)
        c1_2 = self.Conv11_2(c1_2)
        x1 = torch.abs(torch.sub(c1_1, c1_2))

        c2_1 = self.Maxpool(c1_1)
        c2_1 = self.Conv2_1(c2_1)
        c2_2 = self.Maxpool(c1_2)
        c2_2 = self.Conv2_2(c2_2)
        z2 = torch.abs(torch.sub(c2_1, c2_2))
        c2_1 = self.Conv12_1(c2_1)
        c2_2 = self.Conv12_2(c2_2)
        x2 = torch.abs(torch.sub(c2_1, c2_2))

        c3_1 = self.Maxpool(c2_1)
        c3_1 = self.Conv3_1(c3_1)
        c3_2 = self.Maxpool(c2_2)
        c3_2 = self.Conv3_2(c3_2)
        z3 = torch.abs(torch.sub(c3_1, c3_2))
        c3_1 = self.Conv13_1(c3_1)
        c3_2 = self.Conv13_2(c3_2)
        x3 = torch.abs(torch.sub(c3_1, c3_2))

        c4_1 = self.Maxpool(c3_1)
        c4_1 = self.Conv4_1(c4_1)
        c4_2 = self.Maxpool(c3_2)
        c4_2 = self.Conv4_2(c4_2)
        z4 = torch.abs(torch.sub(c4_1, c4_2))
        c4_1 = self.Conv14_1(c4_1)
        c4_2 = self.Conv14_2(c4_2)
        x4 = torch.abs(torch.sub(c4_1, c4_2))

        c5_1 = self.Maxpool(c4_1)
        c5_1 = self.Conv5_1(c5_1)
        c5_2 = self.Maxpool(c4_2)
        c5_2 = self.Conv5_2(c5_2)
        z5 = torch.abs(torch.sub(c5_1, c5_2))
        c5_1 = self.Conv15_1(c5_1)
        c5_2 = self.Conv15_2(c5_2)
        x5 = torch.abs(torch.sub(c5_1, c5_2))

        # encoder2
        c1_1 = self.Conv21_1(c1_1)
        c1_2 = self.Conv21_2(c1_2)
        y1 = torch.abs(torch.sub(c1_1, c1_2))

        # c2_1 = self.Maxpool(c2_1)
        c2_1 = self.Conv22_1(c2_1)
        # c2_2 = self.Maxpool(c2_2)
        c2_2 = self.Conv22_2(c2_2)
        y2 = torch.abs(torch.sub(c2_1, c2_2))

        # c3_1 = self.Maxpool(c3_1)
        c3_1 = self.Conv23_1(c3_1)
        # c3_2 = self.Maxpool(c3_2)
        c3_2 = self.Conv23_2(c3_2)
        y3 = torch.abs(torch.sub(c3_1, c3_2))

        # c4_1 = self.Maxpool(c4_1)
        c4_1 = self.Conv24_1(c4_1)
        # c4_2 = self.Maxpool(c4_2)
        c4_2 = self.Conv24_2(c4_2)
        y4 = torch.abs(torch.sub(c4_1, c4_2))

        # c5_1 = self.Maxpool(c5_1)
        c5_1 = self.Conv25_1(c5_1)
        # c5_2 = self.Maxpool(c5_2)
        c5_2 = self.Conv25_2(c5_2)
        y5 = torch.abs(torch.sub(c5_1, c5_2))

        # decoding
        d5 = self.Up5(x5)
        d25 = self.Up52(y5)
        d05 = self.Up50(z5)
        d25 = torch.cat((d05, d5,d25), dim=1)
        d5 = self.Conv5_1x1(d25)

        # d25 = self.Up52(y5)
        x4 = torch.cat((z4, x4, y4), dim=1)
        x4 = self.Conv4_1x1(x4)
        x3 = torch.cat((z3, x3, y3), dim=1)
        x3 = self.Conv3_1x1(x3)
        x2 = torch.cat((z2, x2, y2), dim=1)
        x2 = self.Conv2_1x1(x2)
        x1 = torch.cat((z1, x1, y1), dim=1)
        x1 = self.Conv1_1x1(x1)

        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        #out5 = self.Conv_5_1x1(d5)
        #out5 = nn.Sigmoid()(out5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        #out4 = self.Conv_4_1x1(d4)
        #out4 = nn.Sigmoid()(out4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        #out3 = self.Conv_3_1x1(d3)
        #out3 = nn.Sigmoid()(out3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)

        return out


if __name__ == "__main__":
    A2016 = torch.randn(1, 3, 256, 256).cuda()
    A2019 = torch.randn(1, 3, 256, 256).cuda()
    model = MDENet(3, 1, ratio=0.5).cuda()
    out_result = model(A2016, A2019)
    summary(model, [(3, 256, 256), (3, 256, 256)])
    flops, params = profile(model, inputs=(A2016, A2019))
    print(flops, params)
