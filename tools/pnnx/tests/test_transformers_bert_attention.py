# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = BertConfig(hidden_size=192, num_attention_heads=16)
        self.attn0 = BertAttention(config0)

        config1 = BertConfig(hidden_size=66, num_attention_heads=6)
        self.attn1 = BertAttention(config1)

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

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_transformers_bert_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_bert_attention.pt inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_bert_attention_pnnx
    b = test_transformers_bert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
