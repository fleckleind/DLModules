# SENet
[Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

SENet: adaptively recalibrateg channel-wise feature responses by explicitly modelling interdependencies between channels.

Squeeze-and-Excitation block: a computational unit constructed for any given transformation $F_{tr}: X\rightarrow U$ with $X\in R^{H^{\prime}\times W^{\prime}\times C^{\prime}}$ and $U\in R^{H\times W\times C}$. Take $F_{tr}$ as a convolutional operator, $V=\[v_1, v_2, \ldots,v_C\]$ as learned set of filter kernels, and the spatial convolution is shown as:
```math
u_c=v_c*X=\sum_{s=1}^{C^{\prime}}v_c^s*x^s
```

## Squeeze: Global Information Embedding
Squeeze Operation: use global average pooling to generate channel-wise statistics, with $z\in R^C$ calculated as:
```math
z_c=F_{sq}(u_c)=\frac{1}{H\times W}\sum_{i=1}^H\sum_{j=1}^W u_c(i,j)
```

## Excitation: Adaptive Recalibration
Excitation Operation: learn nonlinear interaction between channels, and non-mutually-exclusive relationship, with the gating mechanism:
```math
s=F_{ex}(z,W)=\sigma(g(z,W))=\sigma(W_2\delta(W_1z))
```
where $\sigma,\delta$ as $Sigmoid(\cdot)$ and $ReLU(\cdot)$ activation, and the weight sizes are $W_1\in R^{\frac{C}{r}\times C}$, $W_2\in R^{C\times\frac{C}{r}}$, respectively. And the final output is obtained by rescaling the transformation output $U$ with activations:
```math
\tilde{x}_c=F_{scale}(u_c,s_c)=s_c\cdot u_c
```
