# Cross Attention

Cross Attention: allow Query sequence (decoder input) dynamically extract relevant informace from Key-Value sequence (encoder output).

## Mathematical Operation
Cross Attention shares the same operation with self-attention:
```math
Cross\text{ } Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d_k}})V
```
where $Q\in R^{n_q\times d_k}$ linearly projected from the target sequence, while $K\in R^{n_k\times d_k}$ and $V\in R^{n_k\times d_v}$ linearly projected from the source sequence.
