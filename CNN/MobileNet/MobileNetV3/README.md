# MobileNetV3

[Searching for MobileNetV3](http://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)  

MobileNetV3: propose hard non-linearities, h-swish and h-sigmoid activation, and introduce squeeze-and-excitation channel attention module.

## Non-Linearity
$Swish(\cdot)$ activation function is used as a drop-in replacement for $ReLU(\cdot)$, which is defined as 
```math
Swish(x)=x\cdot\sigma(x)
```
while improving accuracy, the $sigmoid(\cdot)$ funciton is more expensive to compute on mobile devices. Referring to piece-wise linear hard analog of $sigmoid(\cdot)$,
```math
h-sigmoid(x)=frac{ReLU6(x+3)}{6}
```
then the hard version of $swish(\cdot)$ becomes
```math
h-swish(x)=x\frac{ReLU6(x+3)}{6}
```

## MobileNetV3 Block
Based on MobileNetV2, MobileNetV3 adds improved squeeze-and-excitation (SE) module between the depthwise convolution and the second pointwise convolution.  

SE module is improves the representation of the model by adaptively weighting the features of each channel. It use global adaptive pooling to squeeze the input feature map into $1\times1\times C$, and two $1\times1$ convolutional layers with $ReLU$ and $h-sigmoid$ activation to get the attention map, with channel compressed as $C/r$ then restored.  

## Lite R-ASPP
To deliver fast semantic segmentation results while mixing features from multiple resolutions, MobileNetV3 propose the Lite R-ASPP (Reduced design of the Atrous Spatial Pyramid Pooling operation). For semantic segmentation, the atrous convolution is applied in the last block of MobileNetV3 backbone.  

The module has three branch based on the output with resolution as 1/16 and 1/8. For branch 1/8, it use $1\times1$ convolution to adjust channel size to the numebers of classes. 
```math
R\_ASPP_{1/8}(x_2)=Conv_{1\times1}(x_2)
```
And branch 1/16 use SE-like structure, with first branch using $1\times1$ convolution, batchnormalize and $ReLU$ to adjust channel size, and second branch using adaptive average pooling, $1\times1$ convolution and $sigmoid$ to get attention map. 
```math
R\_ASPP_{1/16}(x_4)=Sigmoid(Conv{1\times1}(AAP(x_4)))\cdot ReLU(BN(Conv_{1\times1}(x_4)))
```
To add with branch 1/8 feature map, branch 1/16 output use bilinear interpolation for $2\times$ upsampling and $1\times1$ convolution to obtain the same shape. The final output is then upsampled to the original shape via $4\times$ bilinear interpolation.
```math
R\_ASPP(x_2,x_4)=BI_2(R\_ASPP_{1/8}(x_2)+Conv_{1\times1}(BI(R\_ASPP_{1/16}(x_4)))
```
