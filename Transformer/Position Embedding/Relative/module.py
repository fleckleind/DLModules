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
