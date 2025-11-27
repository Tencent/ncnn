# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import LayoutLMConfig
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = LayoutLMConfig(hidden_size=192, num_attention_heads=16, intermediate_size=72, attn_implementation='eager')
        self.attn0 = LayoutLMAttention(config0)

        config1 = LayoutLMConfig(hidden_size=66, num_attention_heads=6, intermediate_size=30, attn_implementation='eager')
        self.attn1 = LayoutLMAttention(config1)

    def forward(self, x, y):
        out0 = self.attn0(x, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None)
        out1 = self.attn1(y, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    a = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_transformers_layoutlm_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_layoutlm_attention.onnx inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_layoutlm_attention_pnnx
    b = test_transformers_layoutlm_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
