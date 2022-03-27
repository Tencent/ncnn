# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

        self.c0 = nn.Parameter(torch.rand(12))
        self.c2 = nn.Parameter(torch.rand(48, 12))

    def forward(self, a0, a1, a2, b0, b1, b2, c1):
        a = torch.addmm(a0, a1, a2)
        b = torch.addmm(b0, b1, b2, beta=1.4, alpha=0.7)
        c = torch.addmm(self.c0, c1, self.c2, beta=1, alpha=1)
        return a, b, c

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(13, 1)
    a1 = torch.rand(13, 16)
    a2 = torch.rand(16, 23)
    b0 = torch.rand(7, 33)
    b1 = torch.rand(7, 26)
    b2 = torch.rand(26, 33)
    c1 = torch.rand(16, 48)

    a = net(a0, a1, a2, b0, b1, b2, c1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1, a2, b0, b1, b2, c1))
    mod.save("test_torch_addmm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_addmm.pt inputshape=[13,1],[13,16],[16,23],[7,33],[7,26],[26,33],[16,48]")

    # pnnx inference
    import test_torch_addmm_pnnx
    b = test_torch_addmm_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
