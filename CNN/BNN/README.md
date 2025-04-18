# BNN
Binary Neural Network (BNN): 
Binarization, a 1-bit quantization with only two possible values as -1(0) or +1, is hardware-friendly properties including memory saving and significant acceleration.

## Foward Propagation
Binarization function is defined as follows, with $b_w, b_a$ as tensor of binary weights (kernel) and activations:
```math
Q_w(w)=\alpha b_w, \quad Q_a(a)=\beta b_a
```
In the literature, the sign function is widely used for $Q_w$ an $Q_a$:
```math
sign(x)=\left\{\begin{aligned}
+1&, x\geq 0 \\
-1&, otherwise
\end{aligned}\right.
```
And the vector multiplication (convolutional process) in forward propagation can be reformulated as follows, with $\odot$ as the inner product (bitwise operation XNOR-Bitcount/Popcount) for vectors:
```math
z=\sigma(Q_w(w)\otimes Q_a(a))=\sigma(\alpha\beta(b_w\odot b_a)
```

## Backward Propagation
Apparently, the binarization function is not differentiable, $sign(0)$, and the derivative value in part of the function vanishes, $sign(x\neq0)$. Straight-Through Estimator (STE) is proposed to address the gradient problem occuring when training deep networks binarized by $sign(\cdot)$.
```math
clip(x,-1,1)=max(-1,min(1,x))
```
Binary Neural Networks

## Reference
Native BNN:  
[Bitwise Neural Networks](https://arxiv.org/pdf/1601.06071)  
[Binarized Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf)  
[BinaryConnect: Training Deep Neural Networks with Binary Weights During Propagations](https://proceedings.neurips.cc/paper_files/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf)  

BNN Optimization:  
[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Network](https://arxiv.org/pdf/1603.05279)  
[XNOR-Net++: Improved Binary Neural Networks](https://arxiv.org/pdf/1909.13863)  
[Towards Accurate Binary Convolutional Neural Network](https://proceedings.neurips.cc/paper_files/paper/2017/file/b1a59b315fc9a3002ce38bbe070ec3f5-Paper.pdf)  
[A Comprehensive Review of Binary Neural Network](https://arxiv.org/pdf/2110.06804)  
[Binary Neural Networks: A Survey](https://arxiv.org/pdf/2004.03333)
