# Lightweight Model




## Compressing Approaches

### Parameter Pruning/Quantization
Parameter pruning and quantizing mainly focus on eliminating the redundancy in the model parameters respectively by removing the redundant/uncritical ones or compressing the parameter space, for example, from the floating-point weights to the integer ones.

### Low-Rank Parameter Factorization
Low-rank parameter factorization applies the matrix/tensor decomposition techniques to estimate the informative parameters using the proxy ones of small size.

### Transferred/Compact Convolutional Filter
The compact convolutional filter based on approaches rely on the carefully-designed structual convolutional filters to reduce the storage and computation complexity.

### Knowledge Distillation
The knowledge distillation methods try to distill a more compact model to reproduce the output of a larger network.
