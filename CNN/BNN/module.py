import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryActivation(nn.Module):
    """
    Binary activation: Bi-Real Approximate Function
    out1: 
        x<-1: (-1)*mask1.type(torch.float32)->-1.0
        x>=-1: (x*x+2*x)*(1-mask1.type(torch.float32))->x*x+2*x
    out2:
        x<-1: out1*mask2.type(torch.float32)->-1.0*1.0=-1.0
        -1<=x<0: out1*mask2.type(torch.float32)->(x*x+2*x)*1.0=x*x+2*x
        x>=0: (-x*x+2*x)*(1-mask2.type(torch.float32))->(-x*x+2*x)
    out3:
        x<-1: out2*mask3.type(torch.float32)->-1.0*1.0=-1.0
        -1<=x<0: out2*mask3.type(torch.float32)->(x*x+2*x)*1.0=x*x+2*x
        0<=x<1: out2*mask3.type(torch.float32)->(-x*x+2*x)*1.0=-x*x+2*x
        x>=1: 1*(1-mask3.type(torch.float32))->1.0
    """
    def __init__(self,):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)  # binary
        mask1, mask2, mask3 = x < -1, x < 0, x < 1  # bool feature map
        # mask.type(float32): (<-1, >=-1)->(1.0, 0.0)
        out1 = (-1)*mask1.type(torch.float32) + (x*x+2*x)*(1-mask1.type(torch.float32))
        # mask.type(float32): (<0, >=0)->(1.0, 0.0)
        out2 = out1*mask2.type(torch.float32) + (-x*x+2*x)*(1-mask2.type(torch.float32))
        # mask.type(float32): (<1, >=1)->(1.0, 0.0)
        out3 = out2*mask3.type(torch.float32) + 1*(1-mask3.type(torch.float32))
        # Polynomial Approximation
        # forward: out_forward(sign), backward: out3(approximation)
        out = out_forward.detach() - out3.detach() + out3
        return out


class LearnableBias(nn.Module):
    """
    Learnable Bias: mitigating information loss in channel wise
    """
    def __init__(self, out_ch):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1), requires_grad=True)

    def forward(self, x):
        # out: (1,c,1,1)->(b,c,h,w)
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    """
    Hard Binary Convolution: binarize weights of convolution
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride, self.padding = stride, padding
        # weight number and convolutional matrix size
        self.number_of_weights = in_ch * out_ch * kernel_size * kernel_size
        self.shape = (out_ch, in_ch, kernel_size, kernel_size)
        # binarization amplify weight, initial weight->0
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight  # float weight
        # setting scaling factor (reduce information loss): ||W_r||/n
        scaling_factor = torch.mean(abs(real_weights), dim=(1, 2, 3), keepdim=True)
        scaling_factor = scaling_factor.detach()  # without gradient computation
        # real_weight->-1/1->binary_weight * scaling_factor: supply information
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)  # between (-1,1)
        # STE: forward: binary weights; backward: clipped weights
        binary_weights = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y


class BiResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BiResBlock, self).__init__()
        self.move0 = LearnableBias(in_ch)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_ch, out_ch, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.move1 = LearnableBias(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.move2 = LearnableBias(out_ch)
        self.downsample = downsample

    def forward(self, x):
        # convolution
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn(out)
        # identity mapping/stride for conv
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        out = self.move2(self.prelu(self.move1(out)))
        return out


# ReActNet: https://github.com/liuzechun/ReActNet
def binaryconv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def binaryconv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class BiMobileBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BiMobileBlock, self).__init__()
        # config preparation
        self.binary_act = BinaryActivation()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        if self.stride != 1:
            self.pooling = nn.AvgPool2d(2, 2)
        # binary depth-wise conv
        self.move11 = LearnableBias(in_ch)
        self.binary_3x3 = nn.Sequential(
            binaryconv3x3(in_ch, in_ch, stride=stride),
            nn.BatchNorm2d(in_ch))
        # RPReLU activation
        self.act1 = nn.Sequential(
            LearnableBias(in_ch), nn.PReLU(in_ch), LearnableBias(in_ch),)
        # binary point-wise conv
        self.move21 = LearnableBias(in_ch)
        if self.in_ch == self.out_ch:
            self.binary_pw = nn.Sequential(
                binaryconv1x1(in_ch, out_ch), nn.BatchNorm2d(out_ch),)
        else:
            self.binary_pw1 = nn.Sequential(
                binaryconv1x1(in_ch, in_ch), nn.BatchNorm2d(in_ch),)
            self.binary_pw2 = nn.Sequential(
                binaryconv1x1(in_ch, in_ch), nn.BatchNorm2d(in_ch),)
        self.act2 = nn.Sequential(
            LearnableBias(out_ch), nn.PReLU(out_ch), nn.BatchNorm2d(out_ch),)

    def forward(self, x):
        out1 = self.binary_3x3(self.binary_act(self.move11(x)))
        if self.stride == 2:
            x = self.pooling(x)
        out1 = self.act1(x + out1)
        # duplication
        out2 = self.binary_act(self.move21(out1))
        if self.in_ch == self.out_ch:
            out2 = self.binary_pw(out2) + out1
        else:
            assert self.out_ch == self.in_ch * 2
            out2_1 = self.binary_pw1(out2) + out1
            out2_2 = self.binary_pw2(out2) + out1
            out2 = torch.cat((out2_1, out2_2), dim=1)
        out2 = self.act2(out2)
        return out2
