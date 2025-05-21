import torch


def precompute_freq_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # sinusoidal rotary encoding for each pair, [dim//2]
    freq = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    # create index for each token, from 0 to seq_len-1
    t = torch.arange(seq_len, device=freq.device)  # [seq_len]
    freq = torch.outer(t, freq).float()  # outer product, [seq_len, dim//2]
    freq_cis = torch.polar(torch.ones_like(freq), freq)  # modulo=1, angle=freq
    return freq_cis
  

def apply_rotary_embedding(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freq_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (batch, seq_len, dim)->(batch, seq_len, dim//2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    # complex field: [batch, seq_len, dim//2]
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    # rotary operation, back to real field, adjust dimension
    # shape: (b,s,d//2)->(b,s,d//2)->(b,s,d//2,2)->(b,s,d)
    xq_out = torch.view_as_real(xq_ * freq_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freq_cis).flatten(2)
    # type_as(): transfer data format, and same device
    return xq_out.type_as(xq), xk_out.type_as(xk)
