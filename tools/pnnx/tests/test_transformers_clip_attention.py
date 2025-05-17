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

from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        text_config0 = CLIPTextConfig(hidden_size=192, num_attention_heads=8, attention_dropout=0.0)
        self.text_attn0 = CLIPAttention(text_config0)

        text_config1 = CLIPTextConfig(hidden_size=66, num_attention_heads=11, attention_dropout=0.0)
        self.text_attn1 = CLIPAttention(text_config1)

        vision_config = CLIPVisionConfig(hidden_size=14, num_attention_heads=2, attention_dropout=0.0)
        self.vision_attn = CLIPAttention(vision_config)

    def forward(self, x, y, z):
        out0, _ = self.text_attn0(x, attention_mask=None, causal_attention_mask=None, output_attentions=True)
        out1, _ = self.text_attn1(y, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        out2, _ = self.vision_attn(z, attention_mask=None, causal_attention_mask=None)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(1, 5, 66)
    z = torch.rand(2, 10, 14)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_transformers_clip_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_clip_attention.pt inputshape=[3,16,192],[1,5,66],[2,10,14]")

    # pnnx inference
    import test_transformers_clip_attention_pnnx
    b = test_transformers_clip_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
