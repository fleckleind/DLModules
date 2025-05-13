# Bi-Real Net
[Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)

Bi-Real Net: connect the real activations to activations, after 1-bit convolution and/or BatchNorm layer and before the sign function, of the consecutive block through an identity shortcut.


## Training Algorithm
Piecewise Polynomial Function: second-order approximation of the sign function,

Magnitude-aware Sign Function: make gradient with respect to the real weight depends on both the sign and the magnitude of the current real weight

Pre-train:
