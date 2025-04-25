import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(SELayer, self).__init__()
        hid_ch = in_ch // reduction
        self.hSwish = HardSwish()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, in_ch, kernel_size=1, bias=False), self.hSwish,)

    def forward(self, x):
        return x * self.fc(self.adaptive_pool(x))


class BottleneckV3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, expand_ratio=6, s_e=True, act_fn=None):
        super(BottleneckV3, self).__init__()
        hid_out = in_ch * expand_ratio
        self.act_fn = HardSwish() if act_fn is None else nn.ReLU(inplace=True)
        self.squeeze_and_excitation = SELayer(hid_out) if s_e else nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hid_out, kernel_size=1), nn.BatchNorm2d(hid_out), self.act_fn,)
        if stride != 1:
            self.conv2 = nn.Sequential(
                nn.Conv2d(hid_out, hid_out, kernel_size=3, stride=stride, padding=1, groups=hid_out),
                nn.BatchNorm2d(hid_out), self.act_fn, self.squeeze_and_excitation,)
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(hid_out, hid_out, kernel_size=3, stride=1, padding="same", dilation=dilation, groups=hid_out),
                nn.BatchNorm2d(hid_out), self.act_fn, self.squeeze_and_excitation, )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid_out, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch),)
        # shortcut: residual connection, if stride==1
        self.residual_buf = True if stride == 1 else False
        self.residualConv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if self.residual_buf:
            return self.residualConv(x) + self.conv3(self.conv2(self.conv1(x)))
        else:
            return self.conv3(self.conv2(self.conv1(x)))


class LiteRASPP(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, num_classes):
        super(LiteRASPP, self).__init__()
        self.conv0 = nn.Conv2d(in_ch1, num_classes, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch2, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch2, out_ch, kernel_size=1), nn.Sigmoid(),)
        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(out_ch, num_classes, kernel_size=1),)
        self.output = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, x2, x4):
        out1 = self.conv0(x2)
        out2, attn = self.conv1(x4), self.conv2(x4)
        out2 = self.conv3(out2 * attn)
        out = self.output(out1 + out2)
        return out
