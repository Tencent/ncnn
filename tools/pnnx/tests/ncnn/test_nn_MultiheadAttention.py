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
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.attention_0_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.attention_0_1 = nn.MultiheadAttention(embed_dim=40, num_heads=4, kdim=30, vdim=20)

        if version.parse(torch.__version__) >= version.parse('1.9'):
            self.attention_1_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            self.attention_1_1 = nn.MultiheadAttention(embed_dim=40, num_heads=4, kdim=30, vdim=20, batch_first=True)

    def forward(self, xq, xk, xv, yq, yk, yv, xmask, ymask):
        x0, _ = self.attention_0_0(xq, xq, xq)
        x1, _ = self.attention_0_0(xq, xk, xv)
        x2, _ = self.attention_0_0(xq, xk, xk, attn_mask=xmask)
        x3, _ = self.attention_0_1(yq, yk, yv, attn_mask=ymask)

        if version.parse(torch.__version__) < version.parse('1.9'):
            return x0, x1, x2, x3

        xq = xq.transpose(0, 1)
        xk = xk.transpose(0, 1)
        xv = xv.transpose(0, 1)
        yq = yq.transpose(0, 1)
        yk = yk.transpose(0, 1)
        yv = yv.transpose(0, 1)

        y0, _ = self.attention_1_0(xq, xq, xq)
        y1, _ = self.attention_1_0(xq, xk, xv)
        y2, _ = self.attention_1_0(xq, xk, xk, attn_mask=xmask)
        y3, _ = self.attention_1_1(yq, yk, yv, attn_mask=ymask)

        return x0, x1, x2, x3, y0, y1, y2, y3

def test():
    torch.set_grad_enabled(False)

    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    xq = torch.rand(20, 1, 64)
    xk = torch.rand(20, 1, 64)
    xv = torch.rand(20, 1, 64)
    yq = torch.rand(15, 1, 40)
    yk = torch.rand(24, 1, 30)
    yv = torch.rand(24, 1, 20)
    xmask = torch.rand(20, 20)
    ymask = torch.rand(4, 15, 24)

    a = net(xq, xk, xv, yq, yk, yv, xmask, ymask)

    # export torchscript
    if version.parse(torch.__version__) >= version.parse('1.12.0'):
        mod = torch.jit.trace(net, (xq, xk, xv, yq, yk, yv, xmask, ymask), check_trace=False)
    else:
        mod = torch.jit.trace(net, (xq, xk, xv, yq, yk, yv, xmask, ymask))
    mod.save("test_nn_MultiheadAttention.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_MultiheadAttention.pt inputshape=[20,1,64],[20,1,64],[20,1,64],[15,1,40],[24,1,30],[24,1,20],[20,20],[4,15,24]")

    # ncnn inference
    import test_nn_MultiheadAttention_ncnn
    b = test_nn_MultiheadAttention_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            print(a0)
            print(b0)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
