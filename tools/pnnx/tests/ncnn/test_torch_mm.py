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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a0, a1):
        a = torch.mm(a0, a1)
        return a

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(23, 14)
    a1 = torch.rand(14, 35)

    a = net(a0, a1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1))
    mod.save("test_torch_mm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_mm.pt inputshape=[23,14],[14,35]")

    # ncnn inference
    import test_torch_mm_ncnn
    b = test_torch_mm_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
