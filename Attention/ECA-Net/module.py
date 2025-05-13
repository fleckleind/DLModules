import torch.nn as nn

class ECA(nn.Module):
    """ ECA module: https://github.com/BangguWu/ECANet/tree/master
        Notes: use the best kernel size 3
    Args:
        in_ch: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_ch, k_size=3):
        super(ECA, self).__init__()
        # k_size = int(abs(math.log(in_ch, 2) + b) / gamma)
        # if k_size % 2:
        #     k_size = k_size
        # else:
        #     k_size = k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
      
