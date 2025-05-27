# Self-Attention
[Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Self-Attention: connects all positions with a constant number of sequentially executed operations, with parallel computation.

## Scaled Dot-Product Attention
Scaled Dot-Product Attention: consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. The attention function is calculated on a set of queries simulataneously, and the matrix of output is defined as
```math
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
```
with $d_k^{-1/2}$ as the scaling factor to amplify extremely small gradients caused by dot products and softmax function.

## Multi-Head Attention
Multi-Head Attention: allows the model to jointly attend to information from different representation subspaces at different positions.
```math
\begin{align}
MultiHead(Q,K,V)&=Concat(head_1,\ldots,head_h)W^O\\
\text{where } head_i &= Attention(QW_i^Q, KW_i^K, VW_i^V)
\end{align}
```
where the projections are parameter matrices $W_i^Q\in R^{d_{model}\times d_k}$, $W_i^K\in R^{d_{model}\times d_k}$, $W_i^V\in R^{d_{model}\times d_v}$, and $W^O\in R^{hd_{v}\times d_{model}}$.
