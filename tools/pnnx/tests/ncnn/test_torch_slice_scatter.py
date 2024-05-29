# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

    def forward(self, x, y, z):
        if version.parse(torch.__version__) >= version.parse('1.13') and version.parse(torch.__version__) < version.parse('2.0'):
            out0 = torch.slice_scatter(x, y, start=6, step=1)
        else:
            out0 = torch.slice_scatter(x, y, start=6)
        out1 = torch.slice_scatter(x, z, dim=1, start=2, end=6, step=1)
        return out0, out1

def test():
    if version.parse(torch.__version__) < version.parse('1.11'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(8, 8)
    y = torch.rand(2, 8)
    z = torch.rand(8, 4)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_slice_scatter.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_slice_scatter.pt inputshape=[8,8],[2,8],[8,4]")

    # ncnn inference
    import test_torch_slice_scatter_ncnn
    b = test_torch_slice_scatter_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
