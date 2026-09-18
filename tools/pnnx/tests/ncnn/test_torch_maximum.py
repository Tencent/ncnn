# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, u, v, r, s, t):
        out0 = torch.maximum(x, y)
        out1 = torch.maximum(y, y)
        out2 = torch.maximum(z, torch.ones_like(z) + 0.1)
        u = F.max_pool2d(u, 1)
        v = F.max_pool2d(v, 1)
        out3 = torch.maximum(u, v)
        out4 = torch.maximum(u, u + 0.1)
        out5 = torch.maximum(u, r)
        out6 = torch.maximum(u, s)
        out7 = torch.maximum(u, t)
        return out0, out1, out2, out3, out4, out5, out6, out7

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)
    u = torch.rand(2, 3, 5, 7)
    v = torch.rand(2, 3, 5, 7)
    r = torch.rand(3, 5, 7)
    s = torch.rand(3, 1, 1)
    t = torch.rand(5, 7)

    a = net(x, y, z, u, v, r, s, t)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, u, v, r, s, t))
    mod.save("test_torch_maximum.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_maximum.pt inputshape=[3,16],[3,16],[5,9,3],[2,3,5,7],[2,3,5,7],[3,5,7],[3,1,1],[5,7]")

    # ncnn inference
    import test_torch_maximum_ncnn
    b = test_torch_maximum_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
