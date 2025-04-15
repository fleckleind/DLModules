import torch
import logging
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_


class ConvEmbed(nn.Module):
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super(ConvEmbed, self).__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        # convolutional token embedding
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (c h) w -> b c h w', h=H, w=W)
        return x


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1,
                 with_cls_token=True, **kwargs):
        super(Attention, self).__init__()
        self.dim, self.num_heads = dim_out, num_heads
        self.scale = self.dim ** -0.5  # \sqrt(d)
        self.stride_kv, self.stride_q = stride_kv, stride_q
        self.with_cls_token = with_cls_token
        # convolutional projection: depth-wise convolution
        self.conv_proj_q = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding_q, stride=stride_q, bias=False, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            Rearrange('b c h w -> b (h w) c'),)
        self.conv_proj_k = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding_kv, stride=stride_kv, bias=False, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            Rearrange('b c h w -> b (h w) c'),)
        self.conv_proj_v = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding_kv, stride=stride_kv, bias=False, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            Rearrange('b c h w -> b (h w) c'),)
        # convolutional projection: point-wise convolution
        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # multi-head self-attention configuration
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_conv(self, x, h, w):
        # q/k/v calculation
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        q, k, v = self.conv_proj_q(x), self.conv_proj_k(x), self.conv_proj_v(x)
        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        return q, k, v

    def forward(self, x, h, w):
        q, k, v = self.forward_conv(x, h, w)
        # multi-head self-attention q/k/v
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        # attn = softmax(q * k / \sqrt(d))
        attn = torch.einsum('b h l k, b h t k -> b h l t', [q, k]) * self.scale
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        # out = attn * v
        x = torch.einsum('b h l t, b h t v -> b h l v', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)', h=self.num_heads)
        # ffn
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hid_features=None, out_features=None,
                 act_layer=nn.GELU, mlp_drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hid_features = hid_features or in_features
        self.fc1 = nn.Linear(in_features, hid_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hid_features, out_features)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class CvT(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 mlp_ratio=4., act_layer=nn.GELU, mlp_drop=0., drop_path=0., norm_layer=nn.LayerNorm, **kwargs):
        super(CvT, self).__init__()
        self.with_cls_token = kwargs['with_cls_token']
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop, **kwargs)
        # randomly dropout some connections in network
        self.drop_path = DropPath(attn_drop) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        # multi-layer perception
        dim_mlp_hid = int(dim_out * mlp_ratio)
        self.mlp = MLP(in_features=dim_out, hid_features=dim_mlp_hid, act_layer=act_layer, mlp_drop=mlp_drop)

    def forward(self, x, h, w):
        attn = self.attn(self.norm1(x), h, w)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, patch_stride=16, patch_padding=0, embed_dim=768,
                 norm_layer=nn.LayerNorm, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, act_layer=nn.GELU,
                 drop_rate=0., drop_path_rate=0., attn_drop=0., proj_drop=0., init='trunc_norm', **kwargs):
        super(VisionTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = ConvEmbed(patch_size=patch_size, in_chans=in_chans, stride=patch_stride,
                                     padding=patch_padding, embed_dim=embed_dim, norm_layer=norm_layer)
        # class token for Transformer
        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        # drop for token position
        self.pos_drop = nn.Dropout(p=drop_rate)
        # drop path rate for different stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # convolutional vision transformer
        blocks = []
        for j in range(depth):
            blocks.append(
                CvT(dim_in=embed_dim, dim_out=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[j],
                    act_layer=act_layer, norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        # params initialization
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    # initial function
    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        # 2D -> 1D
        x = rearrange(x, 'b c h w -> b (h w) c')
        # concatenate class token in batch wise 
        cls_token = None
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        # dropout token position
        x = self.pos_drop(x)
        # convolutional vision transformer
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        # depart x and class token
        if self.cls_token is not None:
            cls_token, x = torch.split(x, [1, H*W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x, cls_token
