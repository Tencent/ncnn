# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import MBartConfig
from transformers.models.mbart.modeling_mbart import MBartAttention

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = MBartConfig(attn_implementation='eager')

        self.attn0 = MBartAttention(embed_dim=192, num_heads=12, config=config)
        self.attn1 = MBartAttention(embed_dim=66, num_heads=6, config=config)

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

    def compare(output, expected):
        return torch.allclose(output, expected, 1e-4, 1e-4)

    return test_model_formats(
        net,
        (x, y),
        a,
        "test_transformers_mbart_attention",
        compare,
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
