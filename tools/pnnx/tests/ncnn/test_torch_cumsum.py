# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
#               2023 Xiaomi Corp.        (author: Fangjun Kuang)
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
        # x - 3d
        # y - 2d
        # z - 1d
        x0 = torch.cumsum(x, dim=0)
        x1 = torch.cumsum(x, dim=1)
        x2 = torch.cumsum(x, dim=2)

        y0 = torch.cumsum(y, dim=0)
        y1 = torch.cumsum(y, dim=1)

        z0 = torch.cumsum(z, dim=0)
        return x0, x1, x2, y0, y1, z0

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 16)
    y = torch.rand(5, 9)
    z = torch.rand(3)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_cumsum.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_cumsum.pt inputshape=[2,3,16],[5,9],[3]")

    # ncnn inference
    import test_torch_cumsum_ncnn
    b = test_torch_cumsum_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
