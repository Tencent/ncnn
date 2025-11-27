# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import LxmertConfig
from transformers.models.lxmert.modeling_lxmert import LxmertSelfAttentionLayer, LxmertCrossAttentionLayer

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = LxmertConfig(hidden_size=192, num_attention_heads=16)
        self.attn0 = LxmertSelfAttentionLayer(config0)

        config1 = LxmertConfig(hidden_size=66, num_attention_heads=6)
        self.attn1 = LxmertCrossAttentionLayer(config1)

    def forward(self, x, y, ctx):
        out0 = self.attn0(x, attention_mask=None)
        out1 = self.attn1(y, ctx_tensor=ctx, ctx_att_mask=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)
    ctx = torch.rand(1, 20, 66)

    a = net(x, y, ctx)

    # export onnx
    torch.onnx.export(net, (x, y, ctx), "test_transformers_lxmert_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_lxmert_attention.onnx inputshape=[3,16,192],[1,5,66],[1,20,66]")

    # pnnx inference
    import test_transformers_lxmert_attention_pnnx
    b = test_transformers_lxmert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
