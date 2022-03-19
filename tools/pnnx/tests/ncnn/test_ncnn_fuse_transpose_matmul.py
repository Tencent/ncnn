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

    def forward(self, a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1, i0, i1, j0, j1, k0, k1, l0, l1):
        a = torch.matmul(a0, a1.transpose(-2, -1))
        b = torch.matmul(b0, b1.transpose(-2, -1))
        c = torch.matmul(c0, c1.transpose(-2, -1))
        d = torch.matmul(d0, d1.transpose(-2, -1))
        e = torch.matmul(e0, e1.transpose(-2, -1))
        f = torch.matmul(f0, f1.transpose(-2, -1))
        g = torch.matmul(g0, g1.transpose(-2, -1))
        h = torch.matmul(h0, h1.transpose(-2, -1))
        i = torch.matmul(i0, i1.transpose(-2, -1))
        j = torch.matmul(j0, j1.transpose(-2, -1))
        k = torch.matmul(k0, k1.transpose(-2, -1))
        l = torch.matmul(l0, l1.transpose(-2, -1))
        return a, b, c, d, e, f, g, h, i, j, k, l

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(14)
    a1 = torch.rand(6, 14)
    b0 = torch.rand(13)
    b1 = torch.rand(7, 4, 13)
    c0 = torch.rand(15)
    c1 = torch.rand(5, 7, 9, 15)
    d0 = torch.rand(23, 14)
    d1 = torch.rand(25, 14)
    e0 = torch.rand(4, 5)
    e1 = torch.rand(10, 40, 5)
    f0 = torch.rand(14, 6)
    f1 = torch.rand(2, 4, 20, 6)
    g0 = torch.rand(10, 23, 14)
    g1 = torch.rand(5, 14)
    h0 = torch.rand(7, 8, 13, 14)
    h1 = torch.rand(35, 14)
    i0 = torch.rand(10, 23, 14)
    i1 = torch.rand(10, 5, 14)
    j0 = torch.rand(10, 13, 18)
    j1 = torch.rand(3, 1, 8, 18)
    k0 = torch.rand(1, 5, 23, 11)
    k1 = torch.rand(8, 1, 9, 11)
    l0 = torch.rand(6, 9, 13, 14)
    l1 = torch.rand(6, 9, 15, 14)

    a = net(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1, i0, i1, j0, j1, k0, k1, l0, l1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1, i0, i1, j0, j1, k0, k1, l0, l1))
    mod.save("test_ncnn_fuse_transpose_matmul.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_transpose_matmul.pt inputshape=[14],[6,14],[13],[7,4,13],[15],[5,7,9,15],[23,14],[25,14],[4,5],[10,40,5],[14,6],[2,4,20,6],[10,23,14],[5,14],[7,8,13,14],[35,14],[10,23,14],[10,5,14],[10,13,18],[3,1,8,18],[1,5,23,11],[8,1,9,11],[6,9,13,14],[6,9,15,14]")

    # ncnn inference
    import test_ncnn_fuse_transpose_matmul_ncnn
    b = test_ncnn_fuse_transpose_matmul_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            print(a0.shape)
            print(b0.shape)
            print(a0)
            print(b0)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
