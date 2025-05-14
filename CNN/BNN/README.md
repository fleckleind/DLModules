# BNN

BNN: a type of neural network that activations/features and weights are 1-bit values in all the hidden layer, except the input and output layers. With binarization, compacting 32-bit to 1-bit values, BNN save the expensive model's storage and reduce the matrix computation costs by using XNOR and popcount operations.

Multiplication Operation: contains a large of floating-point operations, including floating-point multiplication and floating-point addition, leads low-speed performance in neural network inference.

## Forward Propagation
With the basic convolution operation without bias,
```math
Z = I * W
```
the computation steps inside a neural cell along the forward path:
1. $B_I=sign(I)\approx I, \quad B_W=sign(W)\approx W$
2. $Y =\sum_{i=1}^n B_{W_i}B_{I_i} + b$
3. $Z = Activation(Y)$

and the $sign$ function is widely used for binarization:
```math
Sign(x)=
\begin{cases}
+1, &\text{if } x \geq 0,\\
-1, &\text{otherwise}.
\end{cases}
```
Then the expensive matrix multiplication calculation can be replaced as bitwise XNOR and popcount for binary convolution.

## Backward Propagation
### Binarized Neural Networks
Straight-Through Estimator (STE): use approximate sign function to pass gradient in the backward process. 
```math
STE(x)=\frac{\partial Approx(x)}{\partial x}=
\begin{cases}
1, &\text{if } x \geq -1 \text{ and } x \leq 1, \\
0, &\text{otherwise}.
\end{cases}
```
### Bitwise Neural Networks
Bitwise Neural Networks contains two steps to train the BNN model:
1. Real-value networks with weight compression, with weight values in the range of $\[-1,+1\]$.
2. Initialize bitwise neural network with real-valued parameters, and adopt training strategy similar to STE.
