# Transformer
[Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Transformer: based solely on attention mechanisms, and is auto-regressive at each step (consuming the previously generated symbols as additional input when generating the next).

## Model Architrcture
Encoder: $N=6$ identical layers with $d_{model}=512$, consist of multi-head self-attention, position-wise fully connected feed-forward network, residual conenction and layer normalization.

Decoder: $N=6$ identical layers, consist of masked MHA, MHA with queries from encoder and keys/values from previous layer, and FFN.

## Masked MHA
Masked MHA: prevent leftward information flow in the decoder to preserve the auto-regressive property, implemented by scaled dot-product attention by masking out ($-\infty$) all values in the input of the softmax corresponding to illegal connection $masked_{< i}$.

## Position-wise Feed-Forward Networks
The fully connected feed-forward network consists of two linear transformations with a ReLU activation:
```math
FFN(x)=max(0, xW_1+b_1)W_2+b_2
```
while the dimensionality of input and output is $d_{model}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.

## Embeddings and Softmax
Learned Embeddings: convert the input tokens and output tokens to vectors of dimension $d_{model}$, with scaled factors $\sqrt{d_{model}}$.

Learned Linear Transformation and Softmax: convert the decoder output to predicted next-token probabilities.

Weight Matrix: share the same matrix between the two embedding layers and the pre-softmax linear transformation. 
