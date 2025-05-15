# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse('2.1'):
    exit(0)

from transformers import OpenAIGPTConfig
from transformers.models.openai.modeling_openai import Attention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = OpenAIGPTConfig(vocab_size=50000, n_positions=192, n_ctx=192, n_embd=256, n_layer=12, n_head=8)
        self.attn0 = Attention(config0.n_embd, config0.n_positions, config0, scale=True)

        config1 = OpenAIGPTConfig(vocab_size=50000, n_positions=128, n_ctx=128, n_embd=66, n_layer=2, n_head=11)
        self.attn1 = Attention(config1.n_embd, config1.n_positions, config1, scale=True)

    def forward(self, x, y, mask):
        out0 = self.attn0(x, attention_mask=None, head_mask=None)
        out1 = self.attn1(y, attention_mask=mask, head_mask=None)
        return out0[0], out1[0]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 256)
    y = torch.rand(1, 10, 66)
    mask = torch.rand(1, 1, 10, 10)

    a = net(x, y, mask)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask))
    mod.save("test_transformers_openai_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_openai_attention.pt inputshape=[3,16,256],[1,10,66],[1,1,10,10]")

    # pnnx inference
    import test_transformers_openai_attention_pnnx
    b = test_transformers_openai_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
