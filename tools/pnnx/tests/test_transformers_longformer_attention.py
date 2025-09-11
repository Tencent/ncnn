# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = LongformerConfig(hidden_size=192, num_attention_heads=16, attention_window=[4] * 12)
        self.attn0 = LongformerAttention(config0)

    def forward(self, x, mask0):
        is_index_masked = mask0 < 0
        out0 = self.attn0(x, attention_mask=mask0, layer_head_mask=None, is_index_masked=is_index_masked, is_index_global_attn=None, is_global_attn=None)
        return out0[0],

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    mask0 = torch.rand(3, 16)

    a = net(x, mask0)

    # export torchscript
    mod = torch.jit.trace(net, (x, mask0))
    mod.save("test_transformers_longformer_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_longformer_attention.pt inputshape=[3,16,192],[3,16]")

    # pnnx inference
    import test_transformers_longformer_attention_pnnx
    b = test_transformers_longformer_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
