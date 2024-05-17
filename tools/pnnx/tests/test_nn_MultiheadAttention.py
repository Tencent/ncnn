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
        self.attention_0_1 = nn.MultiheadAttention(embed_dim=64, num_heads=8, bias=False, add_bias_kv=False, add_zero_attn=False)
        self.attention_0_2 = nn.MultiheadAttention(embed_dim=64, num_heads=16, bias=True, add_bias_kv=True, add_zero_attn=True)

        self.attention_0_3 = nn.MultiheadAttention(embed_dim=32, num_heads=8, bias=True)

        self.attention_0_4 = nn.MultiheadAttention(embed_dim=40, num_heads=4, kdim=30, vdim=20)
        self.attention_0_5 = nn.MultiheadAttention(embed_dim=40, num_heads=8, kdim=30, vdim=20, bias=False, add_bias_kv=False, add_zero_attn=False)
        self.attention_0_6 = nn.MultiheadAttention(embed_dim=40, num_heads=10, kdim=30, vdim=20, bias=True, add_bias_kv=True, add_zero_attn=True)

        if version.parse(torch.__version__) >= version.parse('1.9'):
            self.attention_1_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            self.attention_1_1 = nn.MultiheadAttention(embed_dim=64, num_heads=8, bias=False, add_bias_kv=False, add_zero_attn=False, batch_first=True)
            self.attention_1_2 = nn.MultiheadAttention(embed_dim=64, num_heads=16, bias=True, add_bias_kv=True, add_zero_attn=True, batch_first=True)

            self.attention_1_3 = nn.MultiheadAttention(embed_dim=32, num_heads=8, bias=True, batch_first=True)

            self.attention_1_4 = nn.MultiheadAttention(embed_dim=40, num_heads=4, kdim=30, vdim=20, batch_first=True)
            self.attention_1_5 = nn.MultiheadAttention(embed_dim=40, num_heads=8, kdim=30, vdim=20, bias=False, add_bias_kv=False, add_zero_attn=False, batch_first=True)
            self.attention_1_6 = nn.MultiheadAttention(embed_dim=40, num_heads=10, kdim=30, vdim=20, bias=True, add_bias_kv=True, add_zero_attn=True, batch_first=True)

    def forward(self, xq, xk, xv, z, zmask, yq, yk, yv, ymask, ymask2):
        x0, x0w = self.attention_0_0(xq, xk, xv)
        x1, x1w = self.attention_0_1(xq, xk, xv)
        x2, x2w = self.attention_0_2(xq, xk, xk)

        x3, _ = self.attention_0_3(z, z, z, need_weights=False)
        x33, _ = self.attention_0_3(z, z, z, attn_mask=zmask)

        x4, x4w = self.attention_0_4(yq, yk, yv)
        x5, x5w = self.attention_0_5(yq, yk, yv, attn_mask=ymask)
        x6, x6w = self.attention_0_6(yq, yk, yv, attn_mask=ymask2)

        if version.parse(torch.__version__) < version.parse('1.9'):
            return x0, x0w, x1, x1w, x2, x2w, x3, x33, x4, x4w, x5, x5w, x6, x6w

        xq = xq.transpose(0, 1)
        xk = xk.transpose(0, 1)
        xv = xv.transpose(0, 1)
        z = z.transpose(0, 1)
        yq = yq.transpose(0, 1)
        yk = yk.transpose(0, 1)
        yv = yv.transpose(0, 1)

        y0, y0w = self.attention_1_0(xq, xk, xv)
        y1, y1w = self.attention_1_1(xq, xk, xv)
        y2, y2w = self.attention_1_2(xq, xk, xk)

        y3, _ = self.attention_1_3(z, z, z)
        if version.parse(torch.__version__) >= version.parse('1.12') and version.parse(torch.__version__) < version.parse('1.13'):
            # HACK pytorch 1.12 breaks 2-dim zmask
            # https://github.com/pytorch/pytorch/issues/97409
            # zmask2 = zmask.reshape(1, 1, 30, 30).expand(1, 8, 30, 30)
            # y33, _ = self.attention_1_3(z, z, z, attn_mask=zmask2)
            # but it produce all nan then, skip test :(
            y33 = y3
        elif version.parse(torch.__version__) >= version.parse('2.0') and version.parse(torch.__version__) < version.parse('2.1'):
            # HACK pytorch 2.0 produce all nan, skip test :(
            y33 = y3
        else:
            y33, _ = self.attention_1_3(z, z, z, attn_mask=zmask)

        y4, y4w = self.attention_1_4(yq, yk, yv)
        y5, y5w = self.attention_1_5(yq, yk, yv, attn_mask=ymask)
        y6, y6w = self.attention_1_6(yq, yk, yv, attn_mask=ymask2)

        return x0, x0w, x1, x1w, x2, x2w, x3, x33, x4, x4w, x5, x5w, x6, x6w, y0, y0w, y1, y1w, y2, y2w, y3, y33, y4, y4w, y5, y5w, y6, y6w

def test():
    torch.set_grad_enabled(False)

    net = Model()
    net.eval()

    torch.manual_seed(0)
    xq = torch.rand(20, 1, 64)
    xk = torch.rand(20, 1, 64)
    xv = torch.rand(20, 1, 64)
    z = torch.rand(30, 1, 32)
    zmask = torch.rand(30, 30)
    yq = torch.rand(15, 1, 40)
    yk = torch.rand(24, 1, 30)
    yv = torch.rand(24, 1, 20)
    ymask = torch.rand(15, 24)
    ymask2 = torch.rand(10, 15, 24)

    a = net(xq, xk, xv, z, zmask, yq, yk, yv, ymask, ymask2)

    # export torchscript
    print(torch.__version__)
    if version.parse(torch.__version__) >= version.parse('1.12.0'):
        mod = torch.jit.trace(net, (xq, xk, xv, z, zmask, yq, yk, yv, ymask, ymask2), check_trace=False)
    else:
        mod = torch.jit.trace(net, (xq, xk, xv, z, zmask, yq, yk, yv, ymask, ymask2))
    mod.save("test_nn_MultiheadAttention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_MultiheadAttention.pt inputshape=[20,1,64],[20,1,64],[20,1,64],[30,1,32],[30,30],[15,1,40],[24,1,30],[24,1,20],[15,24],[10,15,24]")

    # pnnx inference
    import test_nn_MultiheadAttention_pnnx
    b = test_nn_MultiheadAttention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
