# RoPE
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

Rotary Position Embedding (RoPE): encodes the absolute position with a rotation matrix, and incorporates the explicit relative position dependency in self-attention formulation.

## Rotary Position Embedding
The query and key on a 2D plane is defined as
```math
f_q(x_m,m)=(W_qx_m)e^{im\theta}, \quad
f_k(x_n,n)=(W_kx_n)e^{in\theta}
```
then the inner product encoding position information is
```math
g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]
```
where $Re\[\cdot\]$ is the real part of a complex number, and $(W_kx_n)^*$ represents the conjugate complex number of $(W_kx_n)$.

Assume that $R_\alpha$ representing the rotation matrix with angle $\alpha$, $R_\alpha$ has the following properties:
```math
R_\alpha^\top=R_{-\alpha}, \quad
R_\alpha\text{ } R_\beta = R_{\alpha+\beta}
```
then the inner product between $R_\alpha x$ and $R_\beta y$ follows:
```math
\langle R_\alpha x, R_\beta y\rangle=(R_\alpha x)^\top(R_\beta y)=x^\top R_\alpha^\top R_\beta y=x^\top R_{\beta-\alpha} y=\langle x, R_{\beta-\alpha} y\rangle
```

According to Euler formula, the multiplication matrix of query is:
```math
f_q(x_m,m)=
\left(\begin{array}{cc} cos(m\theta) & -sin(m\theta) \\
sin(m\theta) & cos(m\theta)\end{array} \right)
\left(\begin{array}{c} q_m^{(1)} \\ q_m^{(2)} \end{array} \right)
```
and the multiplication matrix of key is:
```math
f_k(x_n,n)=
\left(\begin{array}{cc} cos(n\theta) & -sin(n\theta) \\
sin(n\theta) & cos(n\theta)\end{array} \right)
\left(\begin{array}{c} k_n^{(1)} \\ k_n^{(2)} \end{array} \right)
```
then the inner product $\langle f_q(x_m,m), f_k(x_n,n)\rangle$ is
```math
\left(\begin{array}{cc} q_m^{(1)} & q_m^{(2)} \end{array} \right)
\left(\begin{array}{cc} cos((m-n)\theta) & -sin((m-n)\theta) \\
sin((m-n)\theta) & cos((m-n)\theta)\end{array} \right)
\left(\begin{array}{c} k_n^{(1)} \\ k_n^{(2)} \end{array} \right)
```

## Workflow of Rotoary Transformation
The process of rotary transformation is shown as follows:
1. calculate the query and key for each word embeddings,
2. calculate the rotary position encoding for each token,
3. apply the rotation transformation to query and key at each token position in pairs.


## Reference
[https://zhuanlan.zhihu.com/p/642884818](https://zhuanlan.zhihu.com/p/642884818)
