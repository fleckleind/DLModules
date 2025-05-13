# Bitwise Neural Networks
[Bitwise Neural Networks](https://arxiv.org/pdf/1601.06071)

Bitwise Neural Networks: neural networks with all weight parameters binary-valued, including bias, terms, input, and intermediate hidden layer output signals.


## Feedforward
Foward Propagation Procedure: mathematical definition as follow, 
```math
\begin{align}
a_i^l &= b_i^l+\sum_{j}^{K^{l-1}}w_{i,j}^l\otimes z_j^{l-1} \\
z_i^l &= sign(a_i^l)
\end{align}
```
where $z^l\in B^{K^l}$, $W^l\in B^{K^l\times K^{l-1}}$ and $b^l\in R^{K^l}$ with $B$ as the set of bopolar binaries, $\otimes$ as the bitwise XNOR operation, and $l,j,i$ indicating a layer, input and output units of the layer. And the prediction error $\epsilon$ measures the bit-wise agreement of target vector $t$ and the output units of $L-th$ layer using XNOR as a multiplication operator,
```math
\epsilon=\sum_{i}^{K^{L+1}}(1-t_i\otimes z_i^{L+1})/2
```

## Real-Valued Network
Real-Valued Network: a relaxed version of the corresponding bipolar BNN, takes either bitwise inputs or real-valued inputs ranged between -1 and +1, and constrain the weights to have values in $\[\-1,+1]$ by wrapping them with $tanh$ as weight compression technique.
```math
\begin{align}
a_i^l &= tanh(\bar{b}_i^l)+\sum_j^{K^{l-1}}tanh(\bar{w}_{i,j}^l)\bar{z}_j^{l-1}\\
\bar{z}_i^l &= tanh(a_i^l)
\end{align}
```
where $\bar{W}^l\in R^{K^l\times K^{l-1}}$, $\bar{b}^l\in R^{K^l}$, and $\bar{z}\in R^{K^l}$. The error in a hidden layer is calculated as follow,
```math
\delta_j^l(n)=(\sum_i^{K^{l+1}}tanh(\bar{w}_{i,j}^{l+1}\delta_i^{l+1}(n)))\cdot(1-tanh^2(a_j^l))
```
where the gradients of the parameters in the case of batch learning,
```math
\begin{align}
\nabla\bar{w}_{i,j}^l &= (\sigma_n\delta_i^l(n)\bar{z}_j^{l-1})\cdot(1-tanh^2(\bar{w}_{i,j}^l))\\
\nabla\bar{b}_i^l &= (\sum_n\delta_i^l(n))\cdot(1-tanh^2(\bar{b}_i^l)
\end{align}
```

## Noisy Backpropagation
Initialize all the real-valued parameters $\hat{W}$ and $hat{b}$ from previous section, then setup a sparsity parameter $\lambda$ as the proportion of the zeros after the binarization, $e.g. w_{ij}^l=-1$ if $w_{ij}^l<-\beta$. The errors and gradients are calculated as, 
```math
\begin{align}
\delta_j^l(n) &=\sum_{i}^{K^{l+1}}w_{i,j}^{l+1}\delta_i^{l+1}(n)\\
\nabla\hat{w}_{i,j}^l &=\sum_n\delta_i^l(n)z_j^{l-1}, \quad
\nabla\hat{b}_i^l &=\sum_n\delta_i^l(n).
\end{align}
```
Since the gradients can get too small to update the binary parameters $W$ and $b$, instead update their corresponding real-valued parameters,
```math
\hat{w}_{i,j}^l\leftarrow\hat{w}_{i,j}^l-\nabla\hat{w}_{i,j}^l, \quad
\hat{b}_i^l\leftarrow\hat{b}_i^l-\nabla\hat{b}_i^l
```
At the end of each update, the parameters are binarized with $\beta$.
