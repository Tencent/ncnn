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

        self.w3 = nn.Parameter(torch.rand(16))
        self.b3 = nn.Parameter(torch.rand(16))
        self.w4 = nn.Parameter(torch.rand(12))
        self.b4 = nn.Parameter(torch.rand(12))
        self.w5 = nn.Parameter(torch.rand(32))
        self.b5 = nn.Parameter(torch.rand(32))

    def forward(self, x, y, z):
        x = F.group_norm(x, 4, self.w3, self.b3)
        y = F.group_norm(y, 6, self.w4, self.b4)
        z = F.group_norm(z, 8, self.w5, self.b5, eps=1e-2)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(1, 12, 16)
    z = torch.rand(1, 32, 12, 16)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_group_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_group_norm.pt inputshape=[1,16],[1,12,16],[1,32,12,16]")

    # ncnn inference
    import test_F_group_norm_ncnn
    b = test_F_group_norm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
