# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import MT5Config
from transformers.models.mt5.modeling_mt5 import MT5Attention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = MT5Config(d_model=192, d_kv=64, num_heads=8)
        self.attn = MT5Attention(config, has_relative_attention_bias=True, layer_idx=0)

    def forward(self, x, mask):

        batch_size = x.size(0)
        seq_len = x.size(1)

        cache_position = torch.arange(seq_len)

        out0 = self.attn(x, mask=mask, position_bias=None, use_cache=False, query_length=seq_len, cache_position=cache_position)
        return out0[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    mask = torch.rand(3, 1, 16, 16)

    a = net(x, mask)

    # export torchscript
    mod = torch.jit.trace(net, (x, mask))
    mod.save("test_transformers_mt5_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_mt5_attention.pt inputshape=[3,16,192],[3,1,16,16]")

    # pnnx inference
    import test_transformers_mt5_attention_pnnx
    b = test_transformers_mt5_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
