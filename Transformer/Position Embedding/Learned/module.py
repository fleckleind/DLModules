import torch
import torch.nn as nn

class LearnablePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionEncoding, self).__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # position: (batch_size, seq_len)
        positions = torch.arange(x.size(1)).expand(x.size(0), x.size(1))
        pe = self.positional_embeddings(positions)  # (batch_size, seq_len, d_model)
        return x + pe
      
