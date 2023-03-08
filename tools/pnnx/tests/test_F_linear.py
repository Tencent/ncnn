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

    def forward(self, x, y, z, w0, w1, b1):
        x = F.linear(x, w0, None)
        x = F.linear(x, w1, b1)

        y = F.linear(y, w0, None)
        y = F.linear(y, w1, b1)

        z = F.linear(z, w0, None)
        z = F.linear(z, w1, b1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w0 = torch.rand(12, 16)
    w1 = torch.rand(32, 12)
    b1 = torch.rand(32)

    a0, a1, a2 = net(x, y, z, w0, w1, b1)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w0, w1, b1))
    mod.save("test_F_linear.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_linear.pt inputshape=[1,16],[12,2,16],[1,3,12,16],[12,16],[32,12],[32]")

    # pnnx inference
    import test_F_linear_pnnx
    b0, b1, b2 = test_F_linear_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
