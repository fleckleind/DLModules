# CBAM
[CBAM: Convolutional Block Attention Module](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

CBAM: lightweight and general module, sequentially infer attention maps along two separate dimensions, channel and spatial, then multiply the attention maps to input feature map for adaptive feature refinement.

Convolutional Block Attention Module: given an intermediate feature map $F\in R^{C\times H\times W}$, sequentially infer a 1D channel attention map $M_c\in R^{C\times1\times1}$ and a 2D spatial attention map $M_s\in R^{1\times H\times W}$.

## Channel Attention Module
Channel Sub-Module: utilize both max-pooling outputs $F_{max}^c$ and average-pooling outputs $F_{avg}^c$ with a shared multi-layer perceptron (MLP) owning one hidden layer size as $R^{\frac{C}{r}\times1\times1}$ and $ReLU(\cdot)$ activation.
```math
\begin{align}
M_c(F)&=\sigma(MLP(AvgPool(F))+MLP(MaxPool(F)))\\
&=\sigma(W_1(W_0(F_{avg}^c))+W_1(W_0(F_{max}^c)))
\end{align}
```
And the channel attention module is summarized as follow, with $\otimes$ as element-wise multiplication, brocasting along spatial dimension.
```math
F^{\prime}=M_c(F)\otimes F
```

## Spatial Attention Module
Spatial Sub-Module: apply average-pooling and max-pooling operations along the channel axis, then concatenate $F_{avg}^s\in R^{1\times H\times W}$ and $F_{max}^s\in R^{1\times H\times W}$ to generate an efficient feature descriptor.
```math
\begin{align}
M_s(F)&=\sigma(f^{7\times7}([AvgPool(F);MaxPool(F)]))\\
&=\sigma(f^{7\times7}([F_{avg}^s;F_{max}^s]))
\end{align}
```
And the spatial attention module is summarized as follow, with $\otimes$ as element-wise multiplication, brocasting along channel dimension.
```math
F^{\prime\prime}=M_s(F^{\prime})\otimes F^{\prime}
```
