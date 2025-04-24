import torch.nn as nn

class DepthSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthSepConv, self).__init__()
        self.depConv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),)
        self.poiConv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.poiConv(self.depConv(x))
