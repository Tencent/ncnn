# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import XLMConfig
from transformers.models.xlm.modeling_xlm import MultiHeadAttention

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = XLMConfig(emb_dim=192, n_heads=12)
        self.attn0 = MultiHeadAttention(n_heads=config0.n_heads, dim=config0.emb_dim, config=config0)

        config1 = XLMConfig(emb_dim=66, n_heads=6)
        self.attn1 = MultiHeadAttention(n_heads=config1.n_heads, dim=config1.emb_dim, config=config1)

    def forward(self, x, kv, y, mask0, mask1):
        out0 = self.attn0(x, mask=mask0, kv=kv, cache=None, head_mask=None, output_attentions=True)
        out1 = self.attn1(y, mask=mask1, kv=None, cache=None, head_mask=None, output_attentions=True)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    kv = torch.rand(3, 16, 192)
    y = torch.rand(2, 5, 66)
    mask0 = torch.rand(3, 16)
    mask1 = torch.rand(2, 5)

    a = net(x, kv, y, mask0, mask1)

    def compare(output, expected):
        return torch.allclose(output, expected, 1e-4, 1e-4)

    return test_model_formats(
        net,
        (x, kv, y, mask0, mask1),
        a,
        "test_transformers_xlm_attention",
        compare,
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
