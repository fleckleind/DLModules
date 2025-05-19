# Sinusoidal Position
[Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Sinusoidal Position Encoding: allow the model to extrapolate to sequence lengths longer than the learned positional embeddings encountered during trainings.

## Mathematical Definition
The sine and cosine functions of different frequencies are:
```math
\begin{align}
PE_{(pos, 2i)} &= sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} &= cos(pos/10000^{2i/d_{model}})
\end{align}
```
where $pos$ is the position and $i$ is the dimension, with wavelengths forming a geometric progression from $2\pi$ to $10000\cdot 2\pi$.
