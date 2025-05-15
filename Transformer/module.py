import math
import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728):
        super(SelfAttentionV1, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def __format__(self, x):
        # math.sqrt(self.hidden_dim): gradient explosion
        q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)
        attention_value = torch.matmul(q, k.transpose(-1, -2))
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1)
        output = torch.matmul(attention_weight, v)
        return output


class SelfAttentionV2(nn.Module):
    def __init__(self, dim: int = 728):
        super(SelfAttentionV2, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim * 3)

    def forward(self, x):
        qkv = self.proj(x)
        q, k, v = torch.split(qkv, self.dim, dim=-1)
        attn = torch.softmax(torch.matmul(
            q, k.transpose(-1, -2)) / math.sqrt(self.dim), dim=-1)
        output = attn @ v
        return output


class SelfAttentionV3(nn.Module):
    def __init__(self, dim, dropout_rate=0.1, *args, **kwargs):
        super(SelfAttentionV3, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        qkv = self.proj(x)
        q, k, v = torch.split(qkv, self.dim, dim=-1)
        attn_weight = q @ v.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            attn_weight = attn_weight.masked_fill(
                attention_mask == 0, float('-inf'), )
        attn_weight = self.attn_dropout(torch.softmax(attn_weight, dim=-1))
        output = attn_weight @ v
        return output
