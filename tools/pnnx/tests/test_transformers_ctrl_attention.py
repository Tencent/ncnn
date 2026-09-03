# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers.models.ctrl.modeling_ctrl import MultiHeadAttention

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attn0 = MultiHeadAttention(d_model_size=192, num_heads=16)
        self.attn1 = MultiHeadAttention(d_model_size=66, num_heads=11)

    def forward(self, x, y):
        out0 = self.attn0(x, x, x, mask=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=True)
        out1 = self.attn1(y, y, y, mask=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=True)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    a = net(x, y)

    def compare(output, expected):
        return torch.allclose(output, expected, 1e-4, 1e-4)

    return test_model_formats(
        net,
        (x, y),
        a,
        "test_transformers_ctrl_attention",
        compare,
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
