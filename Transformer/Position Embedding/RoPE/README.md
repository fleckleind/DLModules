# RoPE
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

Rotary Position Embedding (RoPE): encodes the absolute position with a rotation matrix, and incorporates the explicit relative position dependency in self-attention formulation.

## Rotary Position Embedding
The query and key on a 2D plane is defined as
```math
f_q(x_m,m)=(W_qx_m)e^{im\theta}, \quad
f_k(xn,n)=(W_kx_n)e^{in\theta}
```
then the inner product encoding position information is
```math
g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]
```
where $Re\[\cdot\]$ is the real part of a complex number, and $(W_kx_n)^*$ represents the conjugate complex number of $(W_kx_n)$. The $f_{\{q,k\}}$ is written in a multiplication matrix:

