# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import ProphetNetConfig
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = ProphetNetConfig(vocab_size=30522, hidden_size=192, num_encoder_attention_heads=16, num_decoder_attention_heads=16)
        self.attn0 = ProphetNetAttention(config0, num_attn_heads=config0.num_decoder_attention_heads)

        config1 = ProphetNetConfig(vocab_size=30522, hidden_size=66, num_encoder_attention_heads=11, num_decoder_attention_heads=11)
        self.attn1 = ProphetNetAttention(config1, num_attn_heads=config1.num_decoder_attention_heads)

    def forward(self, x, y, mask):
        out0 = self.attn0(x, key_value_states=None, attention_mask=None, layer_head_mask=None, past_key_value=None)
        out1 = self.attn1(y, key_value_states=None, attention_mask=mask, layer_head_mask=None, past_key_value=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(2, 5, 66)
    mask = torch.rand(2, 11, 1, 5)

    a = net(x, y, mask)

    # export onnx
    torch.onnx.export(net, (x, y, mask), "test_transformers_prophetnet_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_prophetnet_attention.onnx inputshape=[3,16,192],[2,5,66],[2,11,1,5]")

    # pnnx inference
    import test_transformers_prophetnet_attention_pnnx
    b = test_transformers_prophetnet_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
