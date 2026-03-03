# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import PegasusConfig
from transformers.models.pegasus.modeling_pegasus import PegasusAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = PegasusConfig(attn_implementation='eager')

        self.attn0 = PegasusAttention(embed_dim=192, num_heads=12, config=config)
        self.attn1 = PegasusAttention(embed_dim=66, num_heads=6, config=config)

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

    # export onnx
    torch.onnx.export(net, (x, y), "test_transformers_pegasus_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_pegasus_attention.onnx inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_pegasus_attention_pnnx
    b = test_transformers_pegasus_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
