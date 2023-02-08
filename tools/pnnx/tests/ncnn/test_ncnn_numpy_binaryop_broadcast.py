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

    def forward(self, x, y, z, w, u, v):
        a = x + y
        b = x - z
        c = x * w
        d = y / z
        e = y + w
        f = z - w
        g = y + x
        h = z - x
        i = w * x
        j = z / y
        k = w + y
        l = w - z
        m = (x - z) * w
        n = (x + y) - (z + w)
        o = x.view(1, 1, 5) + y.view(1, 7, 5) - z
        p = u * y
        q = z / v
        return a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(5)
    y = torch.rand(7, 5)
    z = torch.rand(4, 7, 5)
    w = torch.rand(6, 4, 7, 5)
    u = torch.rand(7, 1)
    v = torch.rand(4, 1, 1)

    a = net(x, y, z, w, u, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, u, v))
    mod.save("test_ncnn_numpy_binaryop_broadcast.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_numpy_binaryop_broadcast.pt inputshape=[5],[7,5],[4,7,5],[6,4,7,5],[7,1],[4,1,1]")

    # ncnn inference
    import test_ncnn_numpy_binaryop_broadcast_ncnn
    b = test_ncnn_numpy_binaryop_broadcast_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
