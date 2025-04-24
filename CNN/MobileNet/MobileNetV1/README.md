# MobileNetV1

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)  

MobileNetV1: depth-wise separable convolutions and two hyper-parameters to trade off between latency and accuracy.

## Depthwise Separable Convolution
Depthwise separable convolution factorizes a standard convolution into a depthwise convolution for filtering and a $1\times1$ pointwise convolution for combining.  
The standard convolution layer is parameterized by convolution kernel $K$ of size $D_K\times D_K\times C_{in}\times C_{out}$, and the output feature map for standard convolution assuming stride one and padding is computed as:
```math
G_{k,l,n}=\sum_{i,j,m}K_{i,j,m,n}\cdot F_{k+i-1, l+j-1, m}
```
with the computational cost of $D_K\cdot D_K\cdot C_{in}\cdot C_{out}\cdot D_F\cdot D_F$, as $D_F\times D_F$ the feature map size.  

Depthwise separable convolution is made up of depthwise convolution to apply a single filter per input channel, and pointwise convolution to create a linear combination of the output of the depthwise layer, with BatchNorm and ReLU for both layers.  
The depthwise convolution is defined as:
```math
\hat{G}_{k,l,m}=\sum_{i,h}\hat{K}_{i,j,m}\cdot F_{k+i-1, l+j-1, m}
```
with kernel size $D_K\times D_K\times C_{in}$, and computational cost $D_K\cdot D_K\cdot C_{in}\cdot D_F\cdot D_F$. To combine the filtering features, a pointwise convolution is needed to create new features, with computational cost $C_{in}\times C_{out}\times D_F\times D_F$.

## Hyper-Parameters: Multiplier
Width Multipler $\alpha$ is used to thin a network uniformly at each layer in the channel dimension, with typical setting of 1, 0.75, 0.5 and 0.25.  

Resolution Multipler $\rho$ is applied to the input image and the internal representation of every layer is subsequently reduced, with setting input resolution of 224, 192, 160 or 128.
