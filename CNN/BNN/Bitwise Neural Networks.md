# Bitwise Neural Networks
[Bitwise Neural Networks](https://arxiv.org/pdf/1601.06071)

BNN: with $z^l$, $W^l$ and $b^l$ belonging to the set of bipolar binaries, and $\otimes$ as the bitwise XNOR operation, the forward propagation procedure is defined as follow,
```math
\begin{align}
a_i^l &= b_i^l+\sum_{j}^{K^{l-1}}w_{i,j}^l\otimes z_j^{l-1} \\
z_i^l &= sign(a_i^l)
\end{align}
```  
