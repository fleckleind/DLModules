# Bi-Real Net
[Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)

Bi-Real Net: connect the real activations to activations, after 1-bit convolution and/or BatchNorm layer and before the sign function, of the consecutive block through an identity shortcut.

## 1-bit CNNs
The binary activation and weight are obtained through a sign function,
```math
a_b/w_b=Sign(a_r/w_r)=\begin{cases}
-1, &\text{if } a_r/w_r<0\\
+1, &\text{otherwise}\end{cases}
```
where $a_r$ and $w_r$ indicate the real activation and the real weight, and the structure flow of original BNN is
```text
sign->1-bit convolution->batch normalization
```
the flow of representational capability is $2^{h\times w\times c}$, $(k_h\times k_w\times k_c)^{h\times w\times c}$, and back to $2^{h\times w\times c}$ via sign funciton, and no change for BatchNorm.

## Bi-Real Net
Bi-Real block indicates the structure as 
```text
sign->1-bit convolution->batch normalization->addition operator
```
with addition operator as the shortcut connection between the input activations and the output feature maps after the batch normalization, the representational capability of each entry in the added activations is $(k_h\times k_w\times k_c)^2$, keeping both real and binary activations.

## Training Algorithm
### Approximation Derivative
A piecewise polynomial function is proposed to use as the closer approximation function:
```math
F(a_r)=\begin{cases}
-1, &\text{if } a_r<-1\\
2a_r+a_r^2, &\text{if } -1\leq a_r<0\\
2a_r-a_r^2, &\text{if } 0\leq a_r<1\\
1, &\text{otherwise}\end{cases}
```
and the derivative of approximation function is:
```math
\frac{\partial F(a_r)}{\partial a_r}=\begin{cases}
2+2a_r, &\text{if } -1\leq a_r<0\\
2-2a_r, &\text{if } 0\leq a_r<1\\
0, &\text{otherwise}\end{cases}
```

### Magnitude-aware Gradient
Bi-Real Net replace the sign function by a magnitude-aware function,
```math
\bar{W}_b^{l,t}=\frac{\lVert W_r^{l,t}\rVert_{1,1}}{\vert W_r^{l,t}\vert}Sign(W_r^{l,t})
```
where $\vert W_r^{l,t}\vert$ denotes the number of channels in $W_r^{l,t}$, and the update of $W_r^l$ becomes
```math
\begin{align}
\bar{W}_b^{l,t+1}&=\bar{W}_b^{l,t}-\eta \frac{\partial L}{\partial \bar{W}_b^{l,t}}\frac{\partial\bar{W}_b^{l,t}}{\partial W_r^{l,t}}\\
&=\bar{W}_b^{l,t}-\eta \frac{\partial L}{\partial A_r^{l+1,t}}\bar{\theta}^{l,t}A_b^l\frac{\partial\bar{W}_b^{l,t}}{\partial W_r^{l,t}}\end{align}
```
where $\bar{\theta}^{l,t}$ is associated with the magnitude of $W_r^{l,t}$, and
```math
\frac{\partial\bar{W}_b^{l,t}}{\partial W_r^{l,t}}\approx
\frac{\lVert W_r^{l,t}\rVert_{1,1}}{\vert W_r^{l,t}\vert}\cdot\frac{\partial Sign(W_r^{l,t})}{\partial W_r^{l,t}}\approx
\frac{\lVert W_r^{l,t}\rVert_{1,1}}{\vert W_r^{l,t}\vert}\cdot1_{\vert W_r^{l,t}\vert <1}
```

### Clip Initialization
Bi-Real Net propose to replace $ReLU$ with $clip(-1,x,1)$ to pre-train the real-valued CNN model, as the activation of the $clip$ function is closer to the sign function than $ReLU$.
