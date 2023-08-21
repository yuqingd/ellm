import math
from einops import rearrange, repeat
import torch
from torch import nn, einsum


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn_map = attn
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return out, attn_map


class ObjectCentricCrossAttention(nn.Module):
    def __init__(self, args, d_input):
        super().__init__()
        self.max_freq = 10.
        self.n_freq_bands = 6
        self.d_slot = args.d_slot
        fourier_channels = self.n_freq_bands * 2 + 1
        self.slots = nn.Parameter(torch.randn(args.n_slot, self.d_slot))
        self.layer = Attention(self.d_slot, context_dim=fourier_channels + d_input)

    def forward(self, data):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        enc_pos = fourier_encode(pos, self.max_freq, self.n_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)
        data = torch.cat((data, enc_pos), dim=-1)

        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.slots, 'n d -> b n d', b=b)
        x, attn_map = self.layer(x, context=data)

        if attn_map.shape[1] > 1:
            attn_map = attn_map[:1]
        attn_map = attn_map * math.sqrt(attn_map.shape[-1])
        return x, attn_map
