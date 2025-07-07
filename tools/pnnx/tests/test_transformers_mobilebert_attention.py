# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import MobileBertConfig
from transformers.models.mobilebert.modeling_mobilebert import MobileBertAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = MobileBertConfig(hidden_size=192, num_attention_heads=16, use_bottleneck=False)
        self.attn0 = MobileBertAttention(config0)

        config1 = MobileBertConfig(hidden_size=66, intra_bottleneck_size=22, num_attention_heads=11, use_bottleneck=True)
        self.attn1 = MobileBertAttention(config1)

    def forward(self, x, y, v, r):
        out0 = self.attn0(x, x, x, layer_input=x, attention_mask=None, head_mask=None)
        out1 = self.attn1(y, y, v, layer_input=r, attention_mask=None, head_mask=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 22)
    v = torch.rand(1, 5, 66)
    r = torch.rand(1, 5, 22)

    a = net(x, y, v, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, v, r))
    mod.save("test_transformers_mobilebert_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_mobilebert_attention.pt inputshape=[3,16,192],[1,5,22],[1,5,66],[1,5,22]")

    # pnnx inference
    import test_transformers_mobilebert_attention_pnnx
    b = test_transformers_mobilebert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
