# ReActNet
[ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions](https://arxiv.org/pdf/2003.03488)

ReActNet: generalize traditional Sign and PReLU functions as RSign and RPReLU for explicit learning of activation distribution, and adopt a distribution loss for similarity between binary and real-valued network.

## 1-bit Convolution
With both weights and activations are binarized to -1 and +1, the computationally heavy operations of floating-point matrix multiplication is replaced by light-weighted bitwise XNOR operations and popcount operations:
```math
X_b * W_b = popcount(XNOR(X_b, W_b))
```
where the weights and activations are binarized as follow,
```math
x_b=sign(x_r)=\begin{cases}
+1, &\text{if } x_r>0\\
-1, &\text{if } x_r\leq 0\end{cases},
w_b=\frac{\lVert W_r\rVert_{l1}}{n}sign(w_r)=\begin{cases}
+\frac{\lVert W_r\rVert_{l1}}{n}, &\text{if } w_r>0\\
-\frac{\lVert W_r\rVert_{l1}}{n}, &\text{if } w_r\leq 0\end{cases}
```

## ReActNet
Based on MobileNetV1, the vanilla $3\times3$ and $1\times1$ convolutions inparallel with shortcuts are replaced as 1-bit convolution. And for downsampling layers whose input and output feature map sizes differ, the outputs of two parallel 1-bit point-wise convolutional blocks are concatenated after the duplicated activation.

## ReAct-PReLU and ReAct-Sign
With the binary convolution choosing values from $\{+1, -1\}$, distribution shift in the input real-valued feature map before the sign function result in a completely different output binary activations.

RSign (ReAct-Sign) is defined with channel-wisely learnable thresholds:
```math
x_i^b=h(x_i^r)=\begin{cases}
+1, &\text{if } x_i^r>\alpha_i\\
-1, &\text{if } x_i^r\leq\alpha_i\end{cases}
```
where $x_i^r$ is real-valued input on the $i$-th channel, and $\alpha_i$ is a learnable coefficient controlling the threshold. RPReLU (ReAct-ReLU) is:
```math
f(x_i)=\begin{cases}
x_i-\gamma_i+\epsilon_i, &\text{if } x_i>\gamma_i\\
\beta_i(x_i-\gamma_i)+\epsilon_i, &\text{if } x_i\leq\gamma_i\end{cases}
```
with $\gamma$ shigting the input distribuiton to find a best point to use $\beta$, and $\epsilon$ shifting the output distribution.

## Distribution Loss
Distributional Loss: binary neural networks can learn similar distributions as real-valued networks, formulated as:
```math
L_{Distribution}=-\frac{1}{n}\sum_c\sum_{i=1}^n p_c^{R_\theta}(X_i)log(\frac{p_c^{B_\theta}(X_i)}{p_c^{R_\theta}(X_i)})
```
which is based on KL divergence between the softmax output $p_c$ of real-valued network $R_\theta$ and binary network $B_\theta$.

## Reference
[ReActNet: official implementation.](https://github.com/liuzechun/ReActNet/tree/master)
