# MobileNetV2

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)  

MobileNetV2: inverted residual structure, with shortcut connections between thin bottleneck layers and removing non-linearities in narrow layers, SSDLite for object detection, and Mobile DeepLabv3 for semantic segmentation.

## Linear Bottleneck
Authors highlighted two properties that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space:  
1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Assuming the manifold of interest is low-dimensional, which can be captured by inserting linear bottleneck layers into convolutional blocks, authors set expansion ratio between the size of the input bottleneck and the inner size.

## Inverted Resiudal
Inverted residual block inserts shortcuts to improve the ability of a gradient to propagate across multiplier layers, with first pointwise convolution expanse channel wise and second pointwise convolution w/o non-linear activation to reduce information loss.  

For a block of size $h\times w$, expansion factor $t$ and kernel size $k$ with $C_{in}$ input channels and $C_{out}$ output chaneels, the total number of multiply add required is $h\cdot w\cdot C_{in}\cdot t(C_{in}+k^2+C_{out}$. The inverted residual blocks use ReLU6 as the non-linearity for its robustness, and set expansion factor as 6.








