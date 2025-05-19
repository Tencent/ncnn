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

from transformers import FunnelConfig
from transformers.models.funnel.modeling_funnel import FunnelRelMultiheadAttention, FunnelAttentionStructure

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = FunnelConfig(d_model=192, n_head=16, d_head=6, attention_type="relative_shift", separate_cls=False)
        self.attn0_structure = FunnelAttentionStructure(config)
        self.attn0 = FunnelRelMultiheadAttention(config, block_index=0)

        torch.nn.init.xavier_uniform_(self.attn0.r_w_bias)
        torch.nn.init.xavier_uniform_(self.attn0.r_kernel)
        torch.nn.init.xavier_uniform_(self.attn0.r_r_bias)
        torch.nn.init.xavier_uniform_(self.attn0.r_s_bias)
        torch.nn.init.xavier_uniform_(self.attn0.seg_embed)

    def forward(self, x, mask0):

        attn_inputs = self.attn0_structure.init_attention_inputs(x, attention_mask=mask0, token_type_ids=None)

        out0 = self.attn0(x, x, x, attention_inputs=attn_inputs, output_attentions=True)
        return out0[0],

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16, 192)

    mask0 = torch.rand(3, 16)

    a = net(x, mask0)

    # export onnx
    torch.onnx.export(net, (x, mask0), "test_transformers_funnel_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_transformers_funnel_attention.onnx inputshape=[3,16,192],[3,16]")

    # pnnx inference
    import test_transformers_funnel_attention_pnnx
    b = test_transformers_funnel_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
