import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryActivation(nn.Module):
    def __init__(self,):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1, mask2, mask3 = x < -1, x < 0, x < 1
        out1 = (-1)*mask1.type(torch.float32) + (x*x+2*x)*(1-mask1.type(torch.float32))
        out2 = out1*mask2.type(torch.float32) + (-x*x+2*x)*(1-mask2.type(torch.float32))
        out3 = out2*mask3.type(torch.float32) + 1*(1-mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out


class LearnableBias(nn.Module):
    def __init__(self, out_ch):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        # out: (1,c,1,1,1)->(b,c,d,h,w)
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv3d, self).__init__()
        self.stride, self.padding = stride, padding
        # weight number and convolutional matrix size
        self.number_of_weights = in_ch * out_ch * kernel_size * kernel_size * kernel_size
        self.shape = (out_ch, in_ch, kernel_size, kernel_size, kernel_size)
        # binarization amplify weight, initial weight->0
        self.weight = nn.Parameter(torch.rand(self.shape)*0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight  # float weight
        scaling_factor = torch.mean(abs(real_weights), dim=(1, 2, 3, 4), keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)  # between (-1,1)
        binary_weights = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights
        out = F.conv3d(x, binary_weights, stride=self.stride, padding=self.padding)
        return out


class HardBinaryTransposedConv3d(nn.Module):
    def __init__(self, chan, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(HardBinaryTransposedConv3d, self).__init__()
        self.stride, self.padding, self.output_padding = stride, padding, output_padding
        self.number_of_weights = chan * chan * kernel_size * kernel_size * kernel_size
        self.shape = (chan, chan, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape)*0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights), dim=(1, 2, 3, 4), keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)  # between (-1,1)
        binary_weights = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights
        out = F.conv_transpose3d(x, binary_weights, stride=self.stride, 
                                 padding=self.padding, output_padding=self.output_padding)
        return out


class BiResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, step=1):
        super(BiResBlock, self).__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        if step == 1:
            self.conv1 = nn.Sequential(
                LearnableBias(in_ch), BinaryActivation(),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_ch), LearnableBias(out_ch), nn.PReLU(out_ch), LearnableBias(out_ch),)
            self.conv2 = nn.Sequential(
                LearnableBias(out_ch), BinaryActivation(),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_ch), LearnableBias(out_ch), nn.PReLU(out_ch), LearnableBias(out_ch),)
        else:
            self.conv1 = nn.Sequential(
                LearnableBias(in_ch), BinaryActivation(),
                HardBinaryConv3d(in_ch, out_ch), nn.BatchNorm3d(out_ch),
                LearnableBias(out_ch), nn.PReLU(out_ch), LearnableBias(out_ch),)
            self.conv2 = nn.Sequential(
                LearnableBias(out_ch), BinaryActivation(),
                HardBinaryConv3d(out_ch, out_ch), nn.BatchNorm3d(out_ch),
                LearnableBias(out_ch), nn.PReLU(out_ch), LearnableBias(out_ch),)
        if self.in_ch != self.out_ch:
            self.resd = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm3d(out_ch),)
        else:
            self.resd = nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.resd(x)
      
