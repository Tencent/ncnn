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

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attention_0 = Attention(embed_dim=64, num_heads=4)
        self.attention_1 = Attention(embed_dim=64, num_heads=8, qkv_bias=False)

    def forward(self, x):
        x = self.attention_0(x)
        x = self.attention_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 20, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_fuse_multiheadattention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_multiheadattention.pt inputshape=[1,20,64]")

    # pnnx inference
    import test_pnnx_fuse_multiheadattention_pnnx
    b = test_pnnx_fuse_multiheadattention_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
