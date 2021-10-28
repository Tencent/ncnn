# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attention_0_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.attention_0_1 = nn.MultiheadAttention(embed_dim=64, num_heads=8, bias=False, add_bias_kv=False, add_zero_attn=False)
        self.attention_0_2 = nn.MultiheadAttention(embed_dim=64, num_heads=16, bias=True, add_bias_kv=True, add_zero_attn=True)

        if torch.__version__ >= '1.9':
            self.attention_1_0 = nn.MultiheadAttention(embed_dim=40, num_heads=4, batch_first=True)
            self.attention_1_1 = nn.MultiheadAttention(embed_dim=40, num_heads=8, bias=False, add_bias_kv=False, add_zero_attn=False, batch_first=True)
            self.attention_1_2 = nn.MultiheadAttention(embed_dim=40, num_heads=10, bias=True, add_bias_kv=True, add_zero_attn=True, batch_first=True)

    def forward(self, xq, xk, xv, yq, yk, yv):
        x0, x0w = self.attention_0_0(xq, xk, xv)
        x1, x1w = self.attention_0_1(xq, xk, xv)
        x2, x2w = self.attention_0_2(xq, xk, xv)

        if torch.__version__ < '1.9':
            return x0, x0w, x1, x1w, x2, x2w

        y0, y0w = self.attention_1_0(yq, yk, yv)
        y1, y1w = self.attention_1_1(yq, yk, yv)
        y2, y2w = self.attention_1_2(yq, yk, yv)

        return x0, x0w, x1, x1w, x2, x2w, y0, y0w, y1, y1w, y2, y2w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    xq = torch.rand(20, 1, 64)
    xk = torch.rand(20, 1, 64)
    xv = torch.rand(20, 1, 64)
    yq = torch.rand(1, 15, 40)
    yk = torch.rand(1, 24, 40)
    yv = torch.rand(1, 24, 40)

    a = net(xq, xk, xv, yq, yk, yv)

    # export torchscript
    mod = torch.jit.trace(net, (xq, xk, xv, yq, yk, yv))
    mod.save("test_nn_MultiheadAttention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_MultiheadAttention.pt inputshape=[20,1,64],[20,1,64],[20,1,64],[1,15,40],[1,24,40],[1,24,40]")

    # pnnx inference
    import test_nn_MultiheadAttention_pnnx
    b = test_nn_MultiheadAttention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
