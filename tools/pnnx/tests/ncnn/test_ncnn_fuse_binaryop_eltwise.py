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

    def forward(self, x0, x1, y0, y1, z0, z1):
        a = x0 + x1 * -1.5
        b = y0 * 0.6 + y1 * 0.4
        c = z0 * 3 + z1
        return a, b, c

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(14)
    x1 = torch.rand(14)
    y0 = torch.rand(23, 14)
    y1 = torch.rand(23, 14)
    z0 = torch.rand(20, 15, 9)
    z1 = torch.rand(20, 15, 9)

    a = net(x0, x1, y0, y1, z0, z1)

    # export torchscript
    mod = torch.jit.trace(net, (x0, x1, y0, y1, z0, z1))
    mod.save("test_ncnn_fuse_binaryop_eltwise.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_binaryop_eltwise.pt inputshape=[14],[14],[23,14],[23,14],[20,15,9],[20,15,9]")

    # ncnn inference
    import test_ncnn_fuse_binaryop_eltwise_ncnn
    b = test_ncnn_fuse_binaryop_eltwise_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
