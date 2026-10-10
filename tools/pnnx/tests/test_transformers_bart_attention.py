# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

import transformers
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartAttention
if version.parse(transformers.__version__) < version.parse('4.53'):
    from transformers.models.bart.modeling_bart import BartSdpaAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = BartConfig(attn_implementation='eager')
        self.attn0 = BartAttention(embed_dim=192, num_heads=12, config=config0)
        if version.parse(transformers.__version__) < version.parse('4.53'):
            self.attn1 = BartSdpaAttention(embed_dim=66, num_heads=6)
        else:
            config1 = BartConfig(attn_implementation='sdpa')
            self.attn1 = BartAttention(embed_dim=66, num_heads=6, config=config1)

    def forward(self, x, y):
        out0 = self.attn0(x, attention_mask=None, key_value_states=None, past_key_value=None)
        out1 = self.attn1(y, attention_mask=None, key_value_states=None, past_key_value=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_transformers_bart_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_bart_attention.pt inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_bart_attention_pnnx
    b = test_transformers_bart_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
