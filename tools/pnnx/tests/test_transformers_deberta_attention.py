# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import DebertaConfig
from transformers.models.deberta.modeling_deberta import DebertaAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = DebertaConfig(hidden_size=192, num_attention_heads=12, intermediate_size=123)
        self.attn0 = DebertaAttention(config0)

        config1 = DebertaConfig(hidden_size=66, num_attention_heads=11, intermediate_size=46)
        self.attn1 = DebertaAttention(config1)

    def forward(self, x, y, mask0, mask1):
        out0 = self.attn0(x, mask0, output_attentions=True, query_states=None, relative_pos=None, rel_embeddings=None)
        out1 = self.attn1(y, mask1, output_attentions=True, query_states=None, relative_pos=None, rel_embeddings=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    mask0 = torch.rand(16, 16)
    mask1 = torch.rand(5, 5)

    a = net(x, y, mask0, mask1)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask0, mask1))
    mod.save("test_transformers_deberta_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_deberta_attention.pt inputshape=[3,16,192],[1,5,66],[16,16],[5,5]")

    # pnnx inference
    import test_transformers_deberta_attention_pnnx
    b = test_transformers_deberta_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
