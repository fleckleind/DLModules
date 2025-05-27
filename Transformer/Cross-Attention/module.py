import math
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim, attn_dropout_rate=0.1, *args, **kwargs):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, target, source):
        q = self.q_proj(target)
        k, v = self.k_proj(source), self.v_proj(source)
        attn_weight = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        attn_weight = self.attn_dropout(torch.softmax(attn_weight, dim=-1))
        output = attn_weight @ v
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_dropout_rate=0.1, *args, **kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        self.head_dim = hidden_dim // num_heads
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, target, source):
        batch, seq_len, _ = target.shape
        q, k, v = self.q_proj(target), self.k_proj(source), self.v_proj(source)
        q_state = q.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        k_state = k.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        v_state = v.view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        attn_weight = q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn_weight = self.attn_dropout(torch.softmax(attn_weight, dim=-1))
        output = torch.matmul(attn_weight, v_state).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)
      
