# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = Qwen3Config(hidden_size=192, num_attention_heads=16, num_key_value_heads=16, q_lora_rank=64, kv_lora_rank=128, attn_implementation='sdpa')
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.attn0 = Qwen3Attention(config, layer_idx=1)

    def forward(self, x, mask0):
        batch_size = x.size(0)
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(x, position_ids)
        out0 = self.attn0(x, position_embeddings=position_embeddings, attention_mask=mask0, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=True)
        return out0[0]

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)

    mask0 = torch.rand(3, 1, 16, 16)

    a = net(x, mask0)

    # export torchscript
    mod = torch.jit.trace(net, (x, mask0))
    mod.save("test_transformers_qwen3_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_qwen3_attention.pt inputshape=[3,16,192],[3,1,16,16]")

    # pnnx inference
    import test_transformers_qwen3_attention_pnnx
    b = test_transformers_qwen3_attention_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
