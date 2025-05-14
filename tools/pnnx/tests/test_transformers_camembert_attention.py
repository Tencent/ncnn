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

from transformers import CamembertConfig
from transformers.models.camembert.modeling_camembert import CamembertAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = CamembertConfig(hidden_size=192, num_attention_heads=16)
        self.attn0 = CamembertAttention(config0)

        config1 = CamembertConfig(hidden_size=66, num_attention_heads=6)
        self.attn1 = CamembertAttention(config1)

    def forward(self, x, y):
        out0 = self.attn0(x, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None)
        out1 = self.attn1(y, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None)
        return out0[0], out1[0]

def test():
    if version.parse(torch.__version__) < version.parse('1.9'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_transformers_camembert_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_camembert_attention.pt inputshape=[3,16,192],[1,5,66]")

    # pnnx inference
    import test_transformers_camembert_attention_pnnx
    b = test_transformers_camembert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
