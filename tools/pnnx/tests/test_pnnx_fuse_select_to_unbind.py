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

    def forward(self, x):
        x0 = torch.select(x, 0, 0)
        x1 = torch.select(x, 0, 1)
        x2 = torch.select(x, 0, 2)
        y0 = torch.select(x, 1, 0)
        y1 = torch.select(x, 1, 1)
        y2 = torch.select(x, 1, 2)
        y3 = torch.select(x, 1, 3)
        z0 = torch.select(x, 2, 0)
        z1 = torch.select(x, 2, 1)
        z2 = torch.select(x, 2, 2)
        z3 = torch.select(x, 2, 3)
        z4 = torch.select(x, 2, 4)

        return x0, x1, x2, y0, y1, y2, y3, z0, z1, z2, z3, z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_fuse_select_to_unbind.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_select_to_unbind.pt inputshape=[3,4,5]")

    # pnnx inference
    import test_pnnx_fuse_select_to_unbind_pnnx
    b = test_pnnx_fuse_select_to_unbind_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
