# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

        self.w0 = nn.Parameter(torch.rand(12, 15))
        self.w1 = nn.Parameter(torch.rand(12, 15))
        self.w2 = nn.Parameter(torch.rand(12, 15))
        self.w3 = nn.Parameter(torch.rand(12, 15))
        self.w4 = nn.Parameter(torch.rand(12, 15))
        self.w5 = nn.Parameter(torch.rand(12, 15))
        self.c0 = nn.Parameter(torch.ones(1))
        self.c1 = nn.Parameter(torch.ones(3) + 0.2)

    def forward(self, x):
        c10, c11, _ = torch.unbind(self.c1)
        x0 = x * 10 + self.c0 - c11
        x = x + self.w0 + x0
        x = x - self.w1 + x0.float()
        x = x * self.w2 + x0
        x = x / self.w3 + x0
        x = x // self.w4 + x0
        if version.parse(torch.__version__) >= version.parse('2.0'):
            x = x % self.w5 + x0
        else:
            x = torch.fmod(x, self.w5) + x0
        y = x.int()
        return x, y & 3, y | 3, y ^ 3, y << 3, y >> 3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 15)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_expression.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_expression.pt inputshape=[12,15]")

    # pnnx inference
    import test_pnnx_expression_pnnx
    b = test_pnnx_expression_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
