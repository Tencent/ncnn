# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers.models.fsmt.modeling_fsmt import Attention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attn0 = Attention(embed_dim=192, num_heads=16)
        self.attn1 = Attention(embed_dim=66, num_heads=11)

    def forward(self, x, y):
        out0 = self.attn0(x, x, attn_mask=None)
        out1 = self.attn1(y, y, attn_mask=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16, 3, 192)
    y = torch.rand(5, 1, 66)

    a = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_transformers_fsmt_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_fsmt_attention.onnx inputshape=[16,3,192],[5,1,66]")

    # pnnx inference
    import test_transformers_fsmt_attention_pnnx
    b = test_transformers_fsmt_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
