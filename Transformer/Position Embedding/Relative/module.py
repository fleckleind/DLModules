import torch
import torch.nn as nn


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        # relative distance: (2*max_len+1, d_model)
        self.relative_pe = nn.Parameter(torch.Tensor(max_len*2+1, d_model))
        nn.init.xavier_uniform_(self.relative_pe)  # initialization

    def forward(self, len_q, len_k):
        # len_q, len_k: align to the shape of (Q @ K.T)
        range_vec_q, range_vec_k = torch.arange(len_q), torch.arange(len_k)
        # distance_mat: (len_q, len_k), [-max_len, max_len]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_len, self.max_len)
        final_mat = distance_mat_clipped + self.max_len  # [0, 2*max_len]
        final_mat = torch.LongTensor(final_mat)  # integer index
        embeddings = self.relative_pe[final_mat]  # (len_q, len_l, d_model)
        return embeddings


class RelativeMHAttention(nn.Module):
    def __init__(self, hid_dim, num_heads, attn_dropout_rate=0.1):
        super(RelativeMHAttention, self).__init__()
        self.max_relative_position = 2
        self.head_dim = hid_dim // num_heads
        self.num_heads, self.hid_dim = num_heads, hid_dim

        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.o_proj = nn.Linear(hid_dim, hid_dim)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)

        self.relative_position_k = RelativePositionEncoding(self.max_relative_position, self.head_dim)
        self.relative_position_v = RelativePositionEncoding(self.max_relative_position, self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # w/o relative positional embedding-k
        q_state1 = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_state1 = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_weight1 = q_state1 @ k_state1.transpose(-2, -1)
        # with relative positional embedding-k
        q_state2 = q.permute(1, 0, 2).contiguous().view(seq_len, batch_size*self.num_heads, self.head_dim)
        k_state2 = self.relative_position_k(seq_len, seq_len)
        attn_weight2 = (q_state2 @ k_state2.transpose(-2, -1)).transpose(0, 1)
        attn_weight2 = attn_weight2.contiguous().view(batch_size, self.num_heads, seq_len, seq_len)
        attn_weight = (attn_weight1 + attn_weight2) / math.sqrt(self.head_dim)
        # attention mask
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, float('-inf'))
        attn_weight = self.attn_dropout(torch.softmax(attn_weight, dim=-1))
        # w/o relative positional embedding-v
        v_state1 = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        output1 = attn_weight @ v_state1
        # with relative positional embedding-v
        v_state2 = self.relative_position_v(seq_len, seq_len)
        output2 = attn_weight.permute(2, 0, 1, 3).contiguous().view(
            seq_len, batch_size*self.num_heads, seq_len) @ v_state2
        output2 = output2.transpose(1, 0).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        output = (output1 + output2).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hid_dim)
        return self.o_proj(output)
