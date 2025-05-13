# ECA-Net
[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf)

ECA: a local cross-channel interaction strategy without dimensionality reduction, efficiently implement via 1D convolution.

Efficient Channel Attention Module: given the aggregated features obtained by global average pooling (GAP), generate channel weights by performing a fast 1D convolution of size $k$, where $k$ is adaptively determined via a mapping of channel dimension $C$.

## Local Cross-Channel Interaction
Capture Local Cross-Channel Interaction: employ a band mateix $W_k$ to learn channel attention, involving $k\times C$ parameters also avoiding complete independence among different groups. The weight $w_i$ of $y_i$ is calculated by only considering interaction between $y_i$ and $k$ neighbors,
```math
w_i=\sigma(\sum_{j=1}^kw_i^j y_i^j), \quad y_i^j\in\Omega_i^k
```
and such strategy can be readily implemented by a fast 1D convolution with kenel size of $k$, with $\Omega_i^k$ indicats the set pf $k$ adjacent channels of $y_i$.
```math
w=\sigma(C1D_k(y))
```

## Converage of Local Cross-Channel Interaction
Referring group convolution involving long range convolutions for high-dimensional channels given the fixed number of groups, the coverage of interaction is proportional to channel dimension $C$, with the simpliest mapping as linear funciton.
```math
C=\phi(k) \leftarrow \gamma*k-b
```
Then a possible solution by extending the linear function to a non-linear one is proposed, $i.e.$
```math
C=\phi(k)=2^{(\gamma*k-b)}
```
Then the kernel size can be adaptively deternined by,
```math
k=\psi(C)=\vert\frac{log_2(C)}{\gamma}+\frac{b}{\gamma}\vert_{odd}
```
