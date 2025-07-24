# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import ReformerConfig
from transformers.models.reformer.modeling_reformer import ReformerAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = ReformerConfig(hidden_size=192, num_attention_heads=12, attention_head_size=16, attn_layers=["local"], lsh_attn_chunk_length=8, is_decoder=False)
        self.attn0 = ReformerAttention(config0, layer_id=0)

        config1 = ReformerConfig(hidden_size=66, num_attention_heads=6, attention_head_size=16, attn_layers=["local"], lsh_attn_chunk_length=8, is_decoder=False)
        self.attn1 = ReformerAttention(config1, layer_id=0)

    def forward(self, x, y, mask):
        out0 = self.attn0(x, attention_mask=None, head_mask=None, num_hashes=None, past_buckets_states=None, use_cache=False, buckets=None)
        out1 = self.attn1(y, attention_mask=mask, head_mask=None, num_hashes=None, past_buckets_states=None, use_cache=False, buckets=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(2, 5, 66)
    mask = torch.rand(2, 5)

    a = net(x, y, mask)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask))
    mod.save("test_transformers_reformer_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_reformer_attention.pt inputshape=[3,16,192],[2,5,66],[2,5]")

    # pnnx inference
    import test_transformers_reformer_attention_pnnx
    b = test_transformers_reformer_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
