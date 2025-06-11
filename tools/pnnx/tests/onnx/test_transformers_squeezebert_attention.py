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

from transformers import SqueezeBertConfig
from transformers.models.squeezebert.modeling_squeezebert import SqueezeBertSelfAttention

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config0 = SqueezeBertConfig(vocab_size=30522, hidden_size=192, embedding_size=768, num_attention_heads=12)
        self.attn0 = SqueezeBertSelfAttention(config0, cin=config0.hidden_size)

        config1 = SqueezeBertConfig(vocab_size=30522, hidden_size=66, embedding_size=768, num_attention_heads=6)
        self.attn1 = SqueezeBertSelfAttention(config1, cin=config1.hidden_size)

    def forward(self, x, y, mask0, mask1):
        out0 = self.attn0(x, attention_mask=mask0, output_attentions=True)
        out1 = self.attn1(y, attention_mask=mask1, output_attentions=True)
        return out0["context_layer"], out1["context_layer"]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 192, 16)
    y = torch.rand(2, 66, 5)
    mask0 = torch.rand(12, 16, 16)
    mask1 = torch.rand(6, 5, 5)

    a = net(x, y, mask0, mask1)

    # export onnx
    torch.onnx.export(net, (x, y, mask0, mask1), "test_transformers_squeezebert_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_squeezebert_attention.onnx inputshape=[3,192,16],[2,66,5],[12,16,16],[6,5,5]")

    # pnnx inference
    import test_transformers_squeezebert_attention_pnnx
    b = test_transformers_squeezebert_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
