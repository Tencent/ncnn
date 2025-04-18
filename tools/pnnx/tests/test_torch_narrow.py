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

    def forward(self, x, y, z):
        out0 = torch.narrow(x, 0, 0, 2)
        out1 = torch.narrow(x, 1, 1, 2)
        out2 = torch.narrow(y, 0, 0, 2)
        out3 = torch.narrow(y, 1, 1, 2)
        out4 = torch.narrow(z, 0, 0, 2)
        out5 = torch.narrow(z, 1, 1, 2)
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 3)
    y = torch.rand(5, 3)
    z = torch.rand(3, 5)
    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_narrow.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_narrow.pt inputshape=[3,3],[5,3],[3,5]")

    # pnnx inference
    import test_torch_narrow_pnnx
    b = test_torch_narrow_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
