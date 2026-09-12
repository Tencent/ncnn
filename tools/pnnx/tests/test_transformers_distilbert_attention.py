# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import DistilBertConfig
from transformers import __version__ as _transformers_version
if version.parse(_transformers_version) >= version.parse('5.0'):
    # transformers 5.x unified SDPA/eager attention; only DistilBertSelfAttention exists
    from transformers.models.distilbert.modeling_distilbert import DistilBertSelfAttention
    MultiHeadSelfAttention = DistilBertSelfAttention
    DistilBertSdpaAttention = DistilBertSelfAttention
else:
    from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, DistilBertSdpaAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = DistilBertConfig(dim=192, n_heads=12)
        self.attn0 = MultiHeadSelfAttention(config0)

        config1 = DistilBertConfig(dim=66, n_heads=11)
        self.attn1 = DistilBertSdpaAttention(config1)

    def forward(self, x, y, mask0, mask1):
        if version.parse(_transformers_version) >= version.parse('5.0'):
            # transformers 5.x forward takes a single hidden_states input and a 4D mask
            out0 = self.attn0(x, attention_mask=mask0[:, None, None, :])
            out1 = self.attn1(y, attention_mask=mask1[:, None, None, :])
        else:
            out0 = self.attn0(x, x, x, mask=mask0, head_mask=None, output_attentions=True)
            out1 = self.attn1(y, y, y, mask=mask1, head_mask=None, output_attentions=False)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    mask0 = torch.rand(3, 16)
    mask1 = torch.rand(1, 5)

    a = net(x, y, mask0, mask1)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask0, mask1))
    mod.save("test_transformers_distilbert_attention.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_transformers_distilbert_attention.pt inputshape=[3,16,192],[1,5,66],[3,16],[1,5]")

    # pnnx inference
    import test_transformers_distilbert_attention_pnnx
    b = test_transformers_distilbert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, mask0, mask1), ["[3,16,192]", "[1,5,66]", "[3,16]", "[1,5]"], "test_transformers_distilbert_attention")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
