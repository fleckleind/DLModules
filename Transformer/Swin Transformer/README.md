# Swin Transformer
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](http://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)

Swin Transformer: a hierarchical Transformer whose representation is computed with shifted windows, limiting self-attention computation to non-overlapping local windows while allowing cross-window connection.

## Overall Architecture
Swin Transformer consists of following blocks (Swin-T):
1. Patch Partition: split an input RGB image into non-overlapping patches by a patch splitting module, with patch size of $4\times4$.
2. Linear Embedding: apply on raw-valued feature to project it to an arbitrary dimension, denoted as $C=96$.
3. Swin Transformer: modified self-attention computation, consisting of a shifted window based MSA module followed by a 2-layer MLP with GELU non-linearity, maintain the numbers of tokens, with block numbers for 4 stages as $[2,2,6,2]$.
4. Patch Merging: concatenate the features of each group of $2\times2$ neighboring patches to reduce the number of tokens, and apply a linear layer to reset the output dimension.

## Shifted Window based Self-Attention
 Self-Attention in Non-Overlapped Windows: compute self-attention within local windows, evenly partition the image in a non-overlapping manner. Suppose each window contains $M\times M$ patches, the computational complexity based on an image of $h\times w$ patches are:
```math
\Omega(MSA)=4hwC^2+2(hw)^2C,\quad \Omega(W-MSA)=4hwC^2+2M^2hwC
```

Shifted Window Partitioning in Successive Blocks: alternates between two partitioning configurations, with first module using a regular window partitioning strategy starting from the top-left pixel and next module displacing the windows by $(\lfloor \frac{M}{2},\frac{M}{2}\rfloor)$ pixels, computed as:
```math
\hat{z}^l=W-MSA(LN(z^{l-1}))+z^{l-1},\quad
z^l = MLP(LN(\hat{z}^l))+\hat{z}^l
```
```math
\hat{z}^{l+1}=W-MSA(LN(z^{l}))+z^{l},\quad
z^{l+1} = MLP(LN(\hat{z}^{l+1}))+\hat{z}^{l+1}
```

Efficient Batch Computation for Shifted Configuration: cyclic-shifting toward the top-left direction, with a batch window composed of non-adjacent sub-windows in feature map, and a masking mechanism to limit self-attention computation to within each sub-window.

Relative Position Bias $B\in R^{M^2\times M^2}$: for each head in computing similarity in computing self-attention,
```math
Attention(Q,K,V)=SoftMax(QK^\top/\sqrt{d}+B)V
```
where $Q,K,V\in R^{M^2\times d}$, with relative position along each axies lying in $[-M+1,M-1]$, and values in $B$ taken from $\hat{B}\in R^{(2M-1)\times (2M-1)}$.

## Reference
[https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
