# BNN
Binary Neural Network (BNN): activations (features) and weights are 1-bit in all hidden layers, except for input and output layers. With binarization, BNN compacts 32-bit to 1-bit, saving memory usage and accelerate model inference, especially for matrix computation.

## Basic Convolution
The basic convolution operation without bias is defined as follow:
```math
Z = I \ast W
```
where $I$, $W$, and $Z$ represent activations, weights and output of the convolution operation with matrix multiplication. It contains a large of floating-point operations, including multiplication and addition.

## Foward Propafation
Compared with 32-bit CNN, BNN adds the binarization steps to the activations $I$ and weights $W$ before convolution. The workflow includes:    
$B_I = sign(I), B_W = sign(W)$; 
$y = \sum_{i=1}^N B_{W_i}B_{I_i} + b$; 
$z = Activation(y)$  
The $sign(\cdot)$ widely used for binarization:
```math
sign(x)=\left\{\begin{aligned}
+1&, x\geq 0 \\
-1&, otherwise
\end{aligned}\right.
```
As multiplication calculation between $B_I$ and $B_W$ is similar as XNOR, then bitwise XNOR and popcount are used to replace expansive matrix multiplication in convolutional operation.

## Backward Propagation
Apparently, the binarization function is not differentiable in $sign(0)$, and the derivative value in part of the function vanishes in $sign(x\neq0)$.  
Straight-Through Estimator (STE) is applied to address the gradient problem occuring when training deep networks binarized by $sign(\cdot)$.
```math
clip(x,-1,1)=max(-1,min(1,x))=\left\{\begin{aligned}
x&, -1\leq x\leq 1 \\
-1&, x<-1 \\
1&, x>1
\end{aligned}\right.
```
Besides, two-step training is also proposed to solve such problem, including the first step training real-value networks with weight compression, and second step for target bitwise neural network.

## BNN Optimization
### Quantization Error Minimization
Scaling Factor

### Gradient Approximation
Maintaining $sign(\cdot)$ function in forward propagation, methods 

###


