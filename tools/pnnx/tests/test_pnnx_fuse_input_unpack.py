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
from typing import List

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        return x + z[1], y[0] + y[1], y[1] - z[0] + z[1] - z[2]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4)
    y0 = torch.rand(2, 3, 4)
    y1 = torch.rand(2, 3, 4)
    z0 = torch.rand(2, 3, 4)
    z1 = torch.rand(2, 3, 4)
    z2 = torch.rand(2, 3, 4)
    y = [y0, y1]
    z = [z0, z1, z2]

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_pnnx_pnnx_fuse_input_unpack.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_pnnx_fuse_input_unpack.pt inputshape=[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4]")

    # pnnx inference
    import test_pnnx_pnnx_fuse_input_unpack_pnnx
    b = test_pnnx_pnnx_fuse_input_unpack_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
