# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import FlaubertConfig
from transformers.models.flaubert.modeling_flaubert import MultiHeadAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = FlaubertConfig()
        self.attn0 = MultiHeadAttention(dim=192, n_heads=16, config=config)
        self.attn1 = MultiHeadAttention(dim=66, n_heads=11, config=config)

    def forward(self, x, y, mask0, mask1):
        out0 = self.attn0(x, mask=mask0, head_mask=None)
        out1 = self.attn1(y, mask=mask1, head_mask=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    mask0 = torch.rand(3, 16, 16)
    mask1 = torch.rand(1, 5, 5)

    a = net(x, y, mask0, mask1)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask0, mask1))
    mod.save("test_transformers_flaubert_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_flaubert_attention.pt inputshape=[3,16,192],[1,5,66],[3,16,16],[1,5,5]")

    # pnnx inference
    import test_transformers_flaubert_attention_pnnx
    b = test_transformers_flaubert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
