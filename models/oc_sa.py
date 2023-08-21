from einops import rearrange
import torch
import torch.nn as nn
import math

from framework.utils.assert_shape import assert_shape

from .mlmha import MLMHA


class ObjectCentricSelfAttention(nn.Module):
    def __init__(self, args, d_obj_latents):
        super().__init__()
        self.d_obj_latents = d_obj_latents
        argsT = args.oc_sa
        self.d_output = argsT.d_model

        self.trafo = MLMHA(argsT, d_embed=self.d_obj_latents)
        self.CLS_token = torch.zeros([self.d_obj_latents], dtype=torch.float32)

    def forward(self, obj_latents):

        bs, n_objs = obj_latents.shape[:2]
        assert_shape(obj_latents, [bs, n_objs, self.d_obj_latents])
        
        self.CLS_token = self.CLS_token.to(obj_latents.device)
        CLS_token_batch = torch.tile(self.CLS_token[None, None, :], [obj_latents.shape[0], 1, 1])

        trafo_input = torch.cat([CLS_token_batch, obj_latents], dim=1)
        output, attn_maps = self.trafo(trafo_input)

        # Take only the CLS token and (query) attn_map
        output = output[0]
        attn_maps = attn_maps[1:, :1, 0, 0]
        attn_maps = rearrange(attn_maps, 'b n -> n b')
        
        attn_maps = attn_maps * math.sqrt(attn_maps.shape[-1])
        return output, attn_maps
