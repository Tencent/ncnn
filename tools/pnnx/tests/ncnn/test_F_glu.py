# Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x0 = F.glu(x, dim=0)

        y0 = F.glu(y, dim=0)
        y1 = F.glu(y, dim=1)

        z0 = F.glu(z, dim=0)
        z1 = F.glu(z, dim=1)
        z2 = F.glu(z, dim=2)
        z3 = F.glu(z, dim=-1)

        w0 = F.glu(w, dim=0)
        w1 = F.glu(w, dim=1)
        w2 = F.glu(w, dim=2)
        w3 = F.glu(w, dim=3)
        w4 = F.glu(w, dim=-1)
        return x0, y0, y1, z0, z1, z2, z3, w0, w1, w2, w3, w4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18)
    y = torch.rand(12, 16)
    z = torch.rand(24, 28, 34)
    w = torch.rand(8, 10, 12, 14)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_F_glu.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_glu.pt inputshape=[18],[12,16],[24,28,34],[8,10,12,14]")

    # ncnn inference
    import test_F_glu_ncnn
    b = test_F_glu_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
