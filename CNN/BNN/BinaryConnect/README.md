# BinaryConnect
[BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://proceedings.neurips.cc/paper_files/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf)

BinaryConnect: regularizer, training a DNN with binary weights during the forward and backward propagations.

## Weights: +1 or -1
BinaryConnect constraints the weights to either +1 or -1 during propagations, thus multiply accumulate operations, the key arithmetic operation, are replaced by simple additions (and subtractions) as fixed-point adders which is less expensive.

## Stochastic Binarization
The very straightforward binarization operation is based on $Sign(\cdot)$:
```math
w_b=
\begin{cases}
+1 & \text{if } x \geq 0, \\
-1 & \text{otherwise} .
\end{cases}
```
where $w_b$ is binarized weight and $w$ real-valued weight. An alternative allowing a finer and more correct averaging process is to binarize stochastically:
```math
w_b=
\begin{case}
+1 & \text{with probability } p=\sigma(w), \\
-1 & \text{with probability } 1-p.
\end{case}
```
and $sigma{\cdot}$ is the hard sigmoid function, far less computationally expensive (both in software and specialized hardware implementations):
```math
\sigma(x)=clip(\frac{x+1}{2}, 0, 1)=max(0, min(1, \frac{x+1}{2}))
```

## Training Process






