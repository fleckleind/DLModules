# CvT
[CvT: Introducing Convolutions to Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)  

Convolutional vision Transformer ([CvT](https://github.com/microsoft/CvT)): considering the benifits of CNNs (local receptive field, shared weights, and spatial subsampling), consists of convolutional token embedding and convolutional Transformer block, realising fewer parameters and lower FLOPs.

## Convolutional Token Embedding
The convolutional token embedding layer, implemented as a convolution and layer normalization, allows progrssive decreasment of token sequence length, modeling local spatial contexts from low-level edges to higher order semantic primitives.  
Suppose $x_i\in\mathbb{R}^{H_{i-1}\times W_{i-1}\times C_{i-1}}$ as input to stage $i$, a 2D convolutional operation $f(\cdot)$ of kenerl size $s\times s$, stride $s-o$, padding $p$ (dealing eith boundary conditions), and output new tokens $f(x_{i-1})\in\mathbb{R}^{H_i\times W_i\times C_i}$.
```math
H_i=\lfloor\frac{H_{i-1}+2p-s}{s-o}+1\rfloor, \quad W_i=\lfloor\frac{W_{i-1}+2p-s}{s-o}+1\rfloor
```

## Convolutional Transformer Block
CvT replaces the original position-wise linear projection for Multi-Head Self-Attention (MHSA) with depth-wise separable convolutions, forming the Convolutional Projection layer.  
According to the official implementation, the workflow of convolutional transformer block is reshaping the dimension of $x_i$ from $(b,h\times w,c)$ to $(b,c,h,w)$, depth-wise separable convolution (dpeth-wise conv2d, batch normalization, and point-wise conv2d) with kernel size $s$, and flattening into 1D $(b,h\times w, c)$ for standard linear MHSA.
```math
x_i^{q/k/v}=Flatten(Conv2d(Reshape2d(x_i), s))
```
To further reduce the computation cost for MHSA, CvT proposed squeezed convolutional projection, where the convolutional projection for key and value has stride as 2 with $4\times$ decreasement.
