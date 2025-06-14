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

        text_config0_sdpa = CLIPTextConfig(hidden_size=192, num_attention_heads=8, attention_dropout=0.0, attn_implementation='sdpa')
        self.text_attn0_sdpa = CLIPAttention(text_config0_sdpa)

        text_config1 = CLIPTextConfig(hidden_size=66, num_attention_heads=11, attention_dropout=0.0)
        self.text_attn1 = CLIPAttention(text_config1)

        text_config1_sdpa = CLIPTextConfig(hidden_size=66, num_attention_heads=11, attention_dropout=0.0, attn_implementation='sdpa')
        self.text_attn1_sdpa = CLIPAttention(text_config1_sdpa)

        text_config2 = CLIPTextConfig(hidden_size=66, num_attention_heads=33, attention_dropout=0.0)
        self.text_attn2 = CLIPAttention(text_config2)

        text_config2_sdpa = CLIPTextConfig(hidden_size=66, num_attention_heads=33, attention_dropout=0.0, attn_implementation='sdpa')
        self.text_attn2_sdpa = CLIPAttention(text_config2_sdpa)

        text_config3 = CLIPTextConfig(hidden_size=66, num_attention_heads=22, attention_dropout=0.0)
        self.text_attn3 = CLIPAttention(text_config3)

        text_config3_sdpa = CLIPTextConfig(hidden_size=66, num_attention_heads=22, attention_dropout=0.0, attn_implementation='sdpa')
        self.text_attn3_sdpa = CLIPAttention(text_config3_sdpa)

        text_config4 = CLIPTextConfig(hidden_size=66, num_attention_heads=6, attention_dropout=0.0)
        self.text_attn4 = CLIPAttention(text_config4)

        text_config4_sdpa = CLIPTextConfig(hidden_size=66, num_attention_heads=6, attention_dropout=0.0, attn_implementation='sdpa')
        self.text_attn4_sdpa = CLIPAttention(text_config4_sdpa)

        vision_config0 = CLIPVisionConfig(hidden_size=14, num_attention_heads=2, attention_dropout=0.0)
        self.vision_attn0 = CLIPAttention(vision_config0)

        vision_config0_sdpa = CLIPVisionConfig(hidden_size=14, num_attention_heads=2, attention_dropout=0.0, attn_implementation='sdpa')
        self.vision_attn0_sdpa = CLIPAttention(vision_config0_sdpa)

    def forward(self, x, y, mask0, casual_mask0, z):
        out0, _ = self.text_attn0(x, attention_mask=None, causal_attention_mask=None, output_attentions=True)
        out0_sdpa, _ = self.text_attn0_sdpa(x, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        out1, _ = self.text_attn1(y, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        out1_sdpa, _ = self.text_attn1_sdpa(y, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        out2, _ = self.text_attn2(y, attention_mask=mask0, causal_attention_mask=None, output_attentions=True)
        out2_sdpa, _ = self.text_attn2_sdpa(y, attention_mask=mask0, causal_attention_mask=None, output_attentions=False)
        out3, _ = self.text_attn3(y, attention_mask=None, causal_attention_mask=casual_mask0, output_attentions=True)
        out3_sdpa, _ = self.text_attn3_sdpa(y, attention_mask=None, causal_attention_mask=casual_mask0, output_attentions=False)
        out4, _ = self.text_attn4(y, attention_mask=mask0, causal_attention_mask=casual_mask0, output_attentions=True)
        out4_sdpa, _ = self.text_attn4_sdpa(y, attention_mask=mask0, causal_attention_mask=casual_mask0, output_attentions=False)
        out5, _ = self.vision_attn0(z, attention_mask=None, causal_attention_mask=None)
        out5_sdpa, _ = self.vision_attn0_sdpa(z, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        return out0, out0_sdpa, out1, out1_sdpa, out2, out2_sdpa, out3, out3_sdpa, out4, out4_sdpa, out5, out5_sdpa

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)
    y = torch.rand(2, 5, 66)
    mask0 = torch.rand(2, 1, 5, 5)
    casual_mask0 = torch.rand(2, 1, 5, 5)
    z = torch.rand(2, 10, 14)

    a = net(x, y, mask0, casual_mask0, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, mask0, casual_mask0, z))
    mod.save("test_transformers_clip_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_clip_attention.pt inputshape=[3,16,192],[2,5,66],[2,1,5,5],[2,1,5,5],[2,10,14]")

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
