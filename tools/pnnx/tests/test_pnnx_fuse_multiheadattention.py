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

from einops import rearrange

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

    def forward(self, x, y):
        a = self.attention_0_0(x)
        a = self.attention_0_1(a)

        b = self.attention_1_0(x)
        b = self.attention_1_1(b, y)

        c = self.attention_2_0(x)
        c = self.attention_2_1(c, y)

        d = self.attention_3(x)
        return a, b, c, d

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 20, 64)
    y = torch.rand(1, 20, 17)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_pnnx_fuse_multiheadattention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_multiheadattention.pt inputshape=[1,20,64],[1,20,17]")

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
