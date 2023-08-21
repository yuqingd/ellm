import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        assert self.demb % 2 == 0, "Positional Embedding dim needs to be an even number!"
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class MHA(nn.Module):
    def __init__(self, args):
        super(MHA, self).__init__()
        self.n_head = args.n_head
        self.d_head = args.d_head
        self.qkv_net = nn.Linear(args.d_model, 3 * self.n_head * self.d_head, bias=False)
        self.dropatt = nn.Dropout(args.dropatt)
        self.scale = 1 / (self.d_head ** 0.5)

    def forward(self, h):
        # [hlen x bsz x n_head x d_head]
        qlen, bsz = h.shape[:2]

        heads = self.qkv_net(h)
        head_q, head_k, head_v = torch.chunk(heads, 3, dim=-1)
        klen = head_k.size(0)
        head_q = head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        head_k = head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        head_v = head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        # qlen x bsz x n_head x d_head
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        return attn_vec, attn_prob


class MLMHA(nn.Module):
    def __init__(self, args, d_embed):
        super(MLMHA, self).__init__()
        self.emb_proj = None
        self.pos_emb = PositionalEmbedding(args.d_model)
        if d_embed != args.d_model:
            self.emb_proj = nn.Linear(d_embed, args.d_model)
        self.drop = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(MHA(args))

    def forward(self, word_emb):
        word_emb = word_emb.transpose(0, 1)
        qlen, bsz, d_emb = word_emb.size()
        klen = qlen

        if self.emb_proj is not None:
            word_emb = self.emb_proj(word_emb)

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)
        core_out = word_emb + pos_emb[-qlen:]

        core_out = self.drop(word_emb)
        for i, layer in enumerate(self.layers):
            core_out, attn_maps = layer(core_out)
        core_out = self.drop(core_out)

        return core_out, attn_maps
