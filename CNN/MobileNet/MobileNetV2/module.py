import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        hid_out = in_ch * expand_ratio
        self.invertConv = nn.Sequential(
            # expansion ratio
            nn.Conv2d(in_ch, hid_out, kernel_size=1), nn.BatchNorm2d(hid_out), nn.ReLU6(inplace=True),
            # depth-wise convolution
            nn.Conv2d(hid_out, hid_out, kernel_size=3, stride=stride, padding=1, groups=hid_out),
            nn.BatchNorm2d(hid_out), nn.ReLU6(inplace=True),
            # point-wise convolution w/o activation
            nn.Conv2d(hid_out, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch),)
        # shortcut: residual connection, if stride==1
        self.residual_buf = True if stride==1 else False
        self.residualConv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if self.residual_buf:
            return self.residualConv(x)+self.invertConv(x)
        else:
            return self.invertConv(x)
