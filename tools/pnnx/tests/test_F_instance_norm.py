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

        self.m3 = torch.rand(12)
        self.v3 = torch.rand(12)
        self.w3 = nn.Parameter(torch.rand(12))
        self.b3 = nn.Parameter(torch.rand(12))
        self.m4 = torch.rand(3)
        self.v4 = torch.rand(3)
        self.w4 = nn.Parameter(torch.rand(3))
        self.b4 = nn.Parameter(torch.rand(3))
        self.m5 = torch.rand(10)
        self.v5 = torch.rand(10)
        self.w5 = nn.Parameter(torch.rand(10))
        self.b5 = nn.Parameter(torch.rand(10))

    def forward(self, x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2):
        x = F.instance_norm(x, m0, v0, w0, b0)
        x = F.instance_norm(x, m0, v0, None, None)
        x = F.instance_norm(x, self.m3, self.v3, self.w3, self.b3)

        y = F.instance_norm(y, m1, v1, w1, b1, eps=1e-3)
        y = F.instance_norm(y, m1, v1, None, None)
        y = F.instance_norm(y, self.m4, self.v4, self.w4, self.b4)

        z = F.instance_norm(z, m2, v2, w2, b2)
        z = F.instance_norm(z, m2, v2, None, None, eps=1e-2)
        z = F.instance_norm(z, self.m5, self.v5, self.w5, self.b5)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)
    m0 = torch.rand(12)
    v0 = torch.rand(12)
    w0 = torch.rand(12)
    b0 = torch.rand(12)
    m1 = torch.rand(3)
    v1 = torch.rand(3)
    w1 = torch.rand(3)
    b1 = torch.rand(3)
    m2 = torch.rand(10)
    v2 = torch.rand(10)
    w2 = torch.rand(10)
    b2 = torch.rand(10)

    a0, a1, a2 = net(x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2))
    mod.save("test_F_instance_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_instance_norm.pt inputshape=[1,12,24],[2,3,12,16],[1,10,12,16,24],[12],[12],[12],[12],[3],[3],[3],[3],[10],[10],[10],[10]")

    # pnnx inference
    import test_F_instance_norm_pnnx
    b0, b1, b2 = test_F_instance_norm_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
