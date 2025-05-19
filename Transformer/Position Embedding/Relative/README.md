# Relative Positional Encoding
[Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155)

Relative Positional Encoding: extend the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements.

## Relation-aware Self-Attention
Suppose the edge between input elements $x_i$ and $x_j$ is represented by vectors $a_{ij}^V$ and $a_{ij}^K$, with attention weight and output as
```math
attn=\frac{x_i W^Q(x_j W^K+a_{ij}^K)^\top}{\sqrt{d_z}}, \quad
out=\sum_{j=1}^n \alpha_{ij}(x_j W^V+a_{ij}^V)
```
where $a_{ij}$ is the softmax activation.

## Relative Position Representation
For linear sequence, edges can capture information about the relative position differences between input elements, and the meximum relative position is clipped to a maximum absolute valur of $k$, with $2k+1$ unique edge labels defined as follow:
```math
a_{ij}^K = w_{clip(j-i, k)}^K, \quad
a_{ij}^V = w_{clip(j-i, k)}^V
```
where $clip$ is defined as $clip(x,k)=max(-k, min(k, x))$.
