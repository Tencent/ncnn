# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from einops import rearrange
from typing import Any, Optional, Tuple, Union
import math

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        _, N, C = x.shape
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.permute((0, 1, 3, 2)))
        attn = F.softmax(attn, dim=-1)

        x = (attn.matmul(v)).permute((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class diffusers_CrossAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class diffusers_AttnProcessor2_0:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class diffusers_CrossAttention(nn.Module):
    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64, dropout=0.0, bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        if version.parse(torch.__version__) >= version.parse('2.0'):
            self.processor = diffusers_AttnProcessor2_0()
        else:
            self.processor = diffusers_CrossAttnProcessor()

    def forward(self, hidden_states, encoder_hidden_states=None):
        return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states)

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key):
        dtype = query.dtype

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

class vit_pytorch_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = 0.

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

class diffusers_AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, bias=True)

    def reshape_heads_to_batch_dim(self, tensor, merge_head_and_batch=True):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        if merge_head_and_batch:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor, unmerge_head_and_batch=True):
        head_size = self.num_heads

        if unmerge_head_and_batch:
            batch_head_size, seq_len, dim = tensor.shape
            batch_size = batch_head_size // head_size

            tensor = tensor.reshape(batch_size, head_size, seq_len, dim)
        else:
            batch_size, _, seq_len, dim = tensor.shape

        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        use_torch_2_0_attn = hasattr(F, "scaled_dot_product_attention")

        query_proj = self.reshape_heads_to_batch_dim(query_proj, merge_head_and_batch=not use_torch_2_0_attn)
        key_proj = self.reshape_heads_to_batch_dim(key_proj, merge_head_and_batch=not use_torch_2_0_attn)
        value_proj = self.reshape_heads_to_batch_dim(value_proj, merge_head_and_batch=not use_torch_2_0_attn)

        if use_torch_2_0_attn:
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query_proj, key_proj, value_proj, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.to(query_proj.dtype)
        else:
            attention_scores = torch.baddbmm(
                torch.empty(
                    query_proj.shape[0],
                    query_proj.shape[1],
                    key_proj.shape[1],
                    dtype=query_proj.dtype,
                    device=query_proj.device,
                ),
                query_proj,
                key_proj.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )
            attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
            hidden_states = torch.bmm(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states, unmerge_head_and_batch=not use_torch_2_0_attn)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attention_0_0 = Attention(embed_dim=64, num_heads=4)
        self.attention_0_1 = Attention(embed_dim=64, num_heads=8, qkv_bias=False)

        self.attention_1_0 = CrossAttention(query_dim=64, heads=4, dim_head=16)
        self.attention_1_1 = CrossAttention(query_dim=64, heads=8, dim_head=8, context_dim=17)

        self.attention_2_0 = diffusers_CrossAttention(query_dim=64, heads=4, dim_head=16)
        self.attention_2_1 = diffusers_CrossAttention(query_dim=64, heads=8, dim_head=8, cross_attention_dim=17)

        self.attention_3 = vit_pytorch_Attention(dim=64, heads=4, dim_head=16)

        self.attention_4_0 = CLIPAttention(embed_dim=64, num_heads=4)
        self.attention_4_1 = CLIPAttention(embed_dim=64, num_heads=4)
        self.mask = nn.Parameter(torch.rand(1, 1, 20, 20))

        self.attention_5 = diffusers_AttentionBlock(channels=64, num_head_channels=16)

    def forward(self, x, y, z):
        a = self.attention_0_0(x)
        a = self.attention_0_1(a)

        b = self.attention_1_0(x)
        b = self.attention_1_1(b, y)

        c = self.attention_2_0(x)
        c = self.attention_2_1(c, y)

        d = self.attention_3(x)

        e = self.attention_4_0(x)
        f = self.attention_4_1(x, causal_attention_mask=self.mask)

        g = self.attention_5(z)

        return a, b, c, d, e, f, g

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 20, 64)
    y = torch.rand(1, 20, 17)
    z = torch.rand(1, 64, 6, 6)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_pnnx_fuse_multiheadattention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_multiheadattention.pt inputshape=[1,20,64],[1,20,17],[1,64,6,6]")

    # pnnx inference
    import test_pnnx_fuse_multiheadattention_pnnx
    b = test_pnnx_fuse_multiheadattention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
