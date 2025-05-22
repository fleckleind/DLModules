# BERT
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)

BERT: pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers, can be fine-tuned with just one additional output layer for downstream tasks without substantial task-specific architecture modifications.

## Model Architecture
BERT's model architecture: multi-layer bidirectional Transformer encoder based on the original implementation, with the number of layers as $L$, the hidden size as $H$, and the number of self-attention heads as $A$.

Input/Output Representations: unambiguously represent both a single sentence and a pair of sentences in one token sequence to make BERT handle a variety of down-stream tasks, WordPiece embedding is used:
1. \[CLS\]: classification token, the first token of every sequence,
2. \[SEP\]: separator token, to differentiate the sentences in pair.

The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings. Denote input embedding as $E$, the final hidden vector of \[CLS\] token as $C\in R^{H}$, and the final hidden vector for the $i^{th}$ input token as $T_i\in R^{H}$.

## Pre-Training BERT
Pre-Training: the model is trained on unlabeled data over different pre-training tasks.

Masked LM (MLM): simply mask 15\% of all WordPiece tokens in each sequence at random, and then predict those masked tokens rather than reconstructing the entire input. To mitigate the \[MASK\] token not appearing during fine-tuning, 80\% tokens are replaced as \[MASK\], 10\% tokens are randomly replaced, and 10\% tokens are unchanged. $T_i$ will be used to predict the original token with cross entropy loss.

Next Sentence Prediction (NSP): a binarized next sentence prediction task that can be trivially generated from any monolingual corpus.

## Fine-Tuning BERT
Fine-Tuning: the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks.

