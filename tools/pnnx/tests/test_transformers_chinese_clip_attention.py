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

from transformers.models.chinese_clip.configuration_chinese_clip import ChineseCLIPTextConfig, ChineseCLIPVisionConfig
from transformers.models.chinese_clip.modeling_chinese_clip import ChineseCLIPTextAttention, ChineseCLIPVisionAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = ChineseCLIPTextConfig(hidden_size=192, num_attention_heads=8, attention_probs_dropout_prob=0.0, max_position_embeddings=64, is_decoder=False)
        self.attn0 = ChineseCLIPTextAttention(config0)

        config1 = ChineseCLIPVisionConfig(hidden_size=12, num_attention_heads=2)
        self.attn1 = ChineseCLIPVisionAttention(config1)

    def forward(self, x, y):
        out0, _ = self.attn0(x, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=True)
        out1, _ = self.attn1(y, output_attentions=True)
        return out0, out1

def test():
    if version.parse(torch.__version__) < version.parse('2.1'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 11, 192)
    y = torch.rand(1, 17, 12)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_transformers_chinese_clip_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_transformers_chinese_clip_attention.pt inputshape=[2,11,192],[1,17,12]")

    # pnnx inference
    import test_transformers_chinese_clip_attention_pnnx
    b = test_transformers_chinese_clip_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
