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

    def forward(self, x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15):
        return (x0 - x1, x1 - x0,
                y0 - y1, y1 - y0,
                y0 - y2, y2 - y0,
                y0 - y3, y3 - y0,
                y1 - y2, y2 - y1,
                y1 - y3, y3 - y1,
                y2 - y3, y3 - y2,
                z0 - z1, z1 - z0,
                z0 - z2, z2 - z0,
                z0 - z3, z3 - z0,
                z0 - z4, z4 - z0,
                z0 - z5, z5 - z0,
                z0 - z6, z6 - z0,
                z0 - z7, z7 - z0,
                z1 - z2, z2 - z1,
                z1 - z3, z3 - z1,
                z1 - z4, z4 - z1,
                z1 - z5, z5 - z1,
                z1 - z6, z6 - z1,
                z1 - z7, z7 - z1,
                z2 - z3, z3 - z2,
                z2 - z4, z4 - z2,
                z2 - z5, z5 - z2,
                z2 - z6, z6 - z2,
                z2 - z7, z7 - z2,
                z3 - z4, z4 - z3,
                z3 - z5, z5 - z3,
                z3 - z6, z6 - z3,
                z3 - z7, z7 - z3,
                z4 - z5, z5 - z4,
                z4 - z6, z6 - z4,
                z4 - z7, z7 - z4,
                z5 - z6, z6 - z5,
                z5 - z7, z7 - z5,
                z6 - z7, z7 - z6,
                w0 - w1, w1 - w0,
                w0 - w2, w2 - w0,
                w0 - w3, w3 - w0,
                w0 - w4, w4 - w0,
                w0 - w5, w5 - w0,
                w0 - w6, w6 - w0,
                w0 - w7, w7 - w0,
                w0 - w8, w8 - w0,
                w0 - w9, w9 - w0,
                w0 - w10, w10 - w0,
                w0 - w11, w11 - w0,
                w0 - w12, w12 - w0,
                w0 - w13, w13 - w0,
                w0 - w14, w14 - w0,
                w0 - w15, w15 - w0,
                w1 - w2, w2 - w1,
                w1 - w3, w3 - w1,
                w1 - w4, w4 - w1,
                w1 - w5, w5 - w1,
                w1 - w6, w6 - w1,
                w1 - w7, w7 - w1,
                w1 - w8, w8 - w1,
                w1 - w9, w9 - w1,
                w1 - w10, w10 - w1,
                w1 - w11, w11 - w1,
                w1 - w12, w12 - w1,
                w1 - w13, w13 - w1,
                w1 - w14, w14 - w1,
                w1 - w15, w15 - w1,
                w2 - w3, w3 - w2,
                w2 - w4, w4 - w2,
                w2 - w5, w5 - w2,
                w2 - w6, w6 - w2,
                w2 - w7, w7 - w2,
                w2 - w8, w8 - w2,
                w2 - w9, w9 - w2,
                w2 - w10, w10 - w2,
                w2 - w11, w11 - w2,
                w2 - w12, w12 - w2,
                w2 - w13, w13 - w2,
                w2 - w14, w14 - w2,
                w2 - w15, w15 - w2,
                w3 - w4, w4 - w3,
                w3 - w5, w5 - w3,
                w3 - w6, w6 - w3,
                w3 - w7, w7 - w3,
                w3 - w8, w8 - w3,
                w3 - w9, w9 - w3,
                w3 - w10, w10 - w3,
                w3 - w11, w11 - w3,
                w3 - w12, w12 - w3,
                w3 - w13, w13 - w3,
                w3 - w14, w14 - w3,
                w3 - w15, w15 - w3,
                w4 - w5, w5 - w4,
                w4 - w6, w6 - w4,
                w4 - w7, w7 - w4,
                w4 - w8, w8 - w4,
                w4 - w9, w9 - w4,
                w4 - w10, w10 - w4,
                w4 - w11, w11 - w4,
                w4 - w12, w12 - w4,
                w4 - w13, w13 - w4,
                w4 - w14, w14 - w4,
                w4 - w15, w15 - w4,
                w5 - w6, w6 - w5,
                w5 - w7, w7 - w5,
                w5 - w8, w8 - w5,
                w5 - w9, w9 - w5,
                w5 - w10, w10 - w5,
                w5 - w11, w11 - w5,
                w5 - w12, w12 - w5,
                w5 - w13, w13 - w5,
                w5 - w14, w14 - w5,
                w5 - w15, w15 - w5,
                w6 - w7, w7 - w6,
                w6 - w8, w8 - w6,
                w6 - w9, w9 - w6,
                w6 - w10, w10 - w6,
                w6 - w11, w11 - w6,
                w6 - w12, w12 - w6,
                w6 - w13, w13 - w6,
                w6 - w14, w14 - w6,
                w6 - w15, w15 - w6,
                w7 - w8, w8 - w7,
                w7 - w9, w9 - w7,
                w7 - w10, w10 - w7,
                w7 - w11, w11 - w7,
                w7 - w12, w12 - w7,
                w7 - w13, w13 - w7,
                w7 - w14, w14 - w7,
                w7 - w15, w15 - w7,
                w8 - w9, w9 - w8,
                w8 - w10, w10 - w8,
                w8 - w11, w11 - w8,
                w8 - w12, w12 - w8,
                w8 - w13, w13 - w8,
                w8 - w14, w14 - w8,
                w8 - w15, w15 - w8,
                w9 - w10, w10 - w9,
                w9 - w11, w11 - w9,
                w9 - w12, w12 - w9,
                w9 - w13, w13 - w9,
                w9 - w14, w14 - w9,
                w9 - w15, w15 - w9,
                w10 - w11, w11 - w10,
                w10 - w12, w12 - w10,
                w10 - w13, w13 - w10,
                w10 - w14, w14 - w10,
                w10 - w15, w15 - w10,
                w11 - w12, w12 - w11,
                w11 - w13, w13 - w11,
                w11 - w14, w14 - w11,
                w11 - w15, w15 - w11,
                w12 - w13, w13 - w12,
                w12 - w14, w14 - w12,
                w12 - w15, w15 - w12,
                w13 - w14, w14 - w13,
                w13 - w15, w15 - w13,
                w14 - w15, w15 - w14,
                x0 - y0, y0 - x0,
                x0 - z0, z0 - x0,
                x0 - w0, w0 - x0,
                y0 - z0, z0 - y0,
                y0 - w0, w0 - y0,
                z0 - w0, w0 - z0)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(5)
    x1 = torch.rand(1)
    y0 = torch.rand(7, 5)
    y1 = torch.rand(1, 5)
    y2 = torch.rand(7, 1)
    y3 = torch.rand(1, 1)
    z0 = torch.rand(4, 7, 5)
    z1 = torch.rand(1, 7, 5)
    z2 = torch.rand(4, 1, 5)
    z3 = torch.rand(4, 7, 1)
    z4 = torch.rand(1, 1, 5)
    z5 = torch.rand(1, 7, 1)
    z6 = torch.rand(4, 1, 1)
    z7 = torch.rand(1, 1, 1)
    w0 = torch.rand(6, 4, 7, 5)
    w1 = torch.rand(1, 4, 7, 5)
    w2 = torch.rand(6, 1, 7, 5)
    w3 = torch.rand(6, 4, 1, 5)
    w4 = torch.rand(6, 4, 7, 1)
    w5 = torch.rand(1, 1, 7, 5)
    w6 = torch.rand(1, 4, 1, 5)
    w7 = torch.rand(1, 4, 7, 1)
    w8 = torch.rand(6, 1, 1, 5)
    w9 = torch.rand(6, 1, 7, 1)
    w10 = torch.rand(6, 4, 1, 1)
    w11 = torch.rand(1, 1, 1, 5)
    w12 = torch.rand(1, 1, 7, 1)
    w13 = torch.rand(1, 4, 1, 1)
    w14 = torch.rand(6, 1, 1, 1)
    w15 = torch.rand(1, 1, 1, 1)

    a = net(x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15)

    # export torchscript
    mod = torch.jit.trace(net, (x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15))
    mod.save("test_ncnn_numpy_binaryop_broadcast.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_numpy_binaryop_broadcast.pt inputshape=[5],[1],[7,5],[1,5],[7,1],[1,1],[4,7,5],[1,7,5],[4,1,5],[4,7,1],[1,1,5],[1,7,1],[4,1,1],[1,1,1],[6,4,7,5],[1,4,7,5],[6,1,7,5],[6,4,1,5],[6,4,7,1],[1,1,7,5],[1,4,1,5],[1,4,7,1],[6,1,1,5],[6,1,7,1],[6,4,1,1],[1,1,1,5],[1,1,7,1],[1,4,1,1],[6,1,1,1],[1,1,1,1]")

    # ncnn inference
    import test_ncnn_numpy_binaryop_broadcast_ncnn
    b = test_ncnn_numpy_binaryop_broadcast_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        # allclose may auto broadcast compare
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
