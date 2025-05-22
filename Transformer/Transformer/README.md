# Transformer
[Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Transformer: a new simple network architecture, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Model Architecture
The encoder maps an input sequence of symbol representations $(x_1,\ldots, x_n)$ to a sequence of continuous representations $z=(z_1,\ldots,z_n)$. Given $z$, the decoder then generates an output sequence $(y_1,\ldots,y_m)$ of symbols one element at a time. The model is auto-regressive at each step, consuming the previously generated symbols as additional input when generating the next.

### Encoder
The encoder is composed of a stack of $N=6$ identical layers, consisting of a multi-head self-attention mechanism and a position-wise fully connected feed-forward network, including residual connection for each sub-layers followed by layer normalization. All sub-layers, as well as the embedding layers, produce outputs of dimension $d_{model}=512$.

### Decoder
The decoder is also composed of a stack of $N=6$ identical layers. Compared to the encoder, the decoder insert a third sub-layer, performing multi-head attention over the output of the encoder stack. Specifically, the queries come from previous decoder layer, and the memory keys and calues come from the output of the encoder.

The self-attention sub-layer in the decoder stack to is modified with masking to prevent positions from attending to subsequent positions, ensuring that the predictions for position $i$ can depend only on the known outputs ar positions less than $i$. The implementation is scaled dot-product attention by masking out (setting to $-\infty$) all values in thee input of the softmax which correspond to illegal connections.

### Position-wise Feed-Forward Networks
The fully connected feed-forward network applied to each position separately and identically consists of two linear transformations with a ReLU activation:
```math
FFN(x)=max(0, xW_1+b_1)W_2+b_2
```
while the dimensionality of input and output is $d_{model}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.

### 
