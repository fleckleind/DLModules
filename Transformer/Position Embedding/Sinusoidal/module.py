import torch
import torch.nn as nn

class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super(SinPositionEncoding, self).__init__()
        pe = torch.zeros(max_sequence_length, d_model)  # (max_seq, d_model)
        # get position index, from 0 to max_sequence_length-1: (max_seq, 1)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        # -1/10000^{2i/d_{model}}: exp(-2i/d_{model}*ln(10000))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # odd position
        pe[:, 1::2] = torch.cos(position * div_term)  # even position
        pe = pe.unsqueeze(0)  # (1, max_seq, d_model)
        self.register_buffer('pe', pe)  # w/o gradient

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
      
