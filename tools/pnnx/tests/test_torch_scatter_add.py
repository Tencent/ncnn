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

    def forward(self, x, y0, y1, z):
        out0 = torch.scatter_add(x, 0, y0, z)
        out1 = torch.scatter_add(x, 0, y1, z)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(13, 15)
    y0 = torch.randint(10, (1, 15), dtype=torch.long)
    y1 = torch.randint(10, (12, 15), dtype=torch.long)
    z = torch.rand(12, 15)

    a = net(x, y0, y1, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y0, y1, z))
    mod.save("test_torch_scatter_add.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_scatter_add.pt inputshape=[13,15],[1,15]i64,[12,15]i64,[12,15]")

    # pnnx inference
    import test_torch_scatter_add_pnnx
    b = test_torch_scatter_add_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
