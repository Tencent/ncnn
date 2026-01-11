# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, Conv1D

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = GPT2Config(hidden_size=192, num_attention_heads=8, scale_attn_weights=True, attn_implementation='eager')
        self.attn0 = GPT2Attention(config)

    def forward(self, x, mask0):
        out0 = self.attn0(x, attention_mask=mask0, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=True)
        return out0[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)

    mask0 = torch.rand(3, 8, 16, 16)

    a = net(x, mask0)

    # export onnx
    torch.onnx.export(net, (x, mask0), "test_transformers_gpt2_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_gpt2_attention.onnx inputshape=[3,16,192],[3,8,16,16]")

    # pnnx inference
    import test_transformers_gpt2_attention_pnnx
    b = test_transformers_gpt2_attention_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
