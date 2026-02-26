# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertAttention, AlbertSdpaAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = AlbertConfig(hidden_size=192, num_attention_heads=12)
        self.attn0 = AlbertAttention(config0)

        config1 = AlbertConfig(hidden_size=66, num_attention_heads=3)
        self.attn1 = AlbertSdpaAttention(config1)

    def forward(self, x, y):
        out0, _ = self.attn0(x, attention_mask=None, head_mask=None, output_attentions=True)
        out1, _ = self.attn1(y, attention_mask=None, head_mask=None, output_attentions=True)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_transformers_albert_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_albert_attention.pt inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_albert_attention_pnnx
    b = test_transformers_albert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
