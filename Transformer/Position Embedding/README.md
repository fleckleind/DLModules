# Position Embedding

Position Encoding: inject some information about the relative or absolute position of the tokens in the sequence.

Positional Embedding: add to the input embeddings at the bottoms of the encoder and decoder stacks, having the same dimension $d_{model}$ as the input embeddings.

## Absolute Positional Encoding
Abosolute Positional Encoding: assign a unique positional embedding for each position of the input embeddings, representing the absolute position of elements, including sinusoidal and learned encoding.

## Relative Positional Encoding
Relative Positional Encoding: focus on the relativly positional relationship, representing the distance between elements.
