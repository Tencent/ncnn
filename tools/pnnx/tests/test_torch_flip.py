# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        # 1D
        x0 = torch.flip(x, [0])
        # 2D
        y0 = torch.flip(y, [0])
        y1 = torch.flip(y, [1])
        y2 = torch.flip(y, [-2, -1])
        # 3D
        z0 = torch.flip(z, [0])
        z1 = torch.flip(z, [1])
        z2 = torch.flip(z, [2])
        z3 = torch.flip(z, [0, 1])
        z4 = torch.flip(z, [0, 2])
        z5 = torch.flip(z, [1, 2])
        z6 = torch.flip(z, [0, 1, 2])
        # 4D
        w0 = torch.flip(w, [-1])
        w1 = torch.flip(w, [-2])
        w2 = torch.flip(w, [-3])
        w3 = torch.flip(w, [-4])
        w4 = torch.flip(w, [0, 1])
        w5 = torch.flip(w, [0, 2])
        w6 = torch.flip(w, [0, 3])
        w7 = torch.flip(w, [1, 2])
        w8 = torch.flip(w, [1, 3])
        w9 = torch.flip(w, [2, 3])
        w10 = torch.flip(w, [0, 1, 2])
        w11 = torch.flip(w, [0, 1, 3])
        w12 = torch.flip(w, [0, 2, 3])
        w13 = torch.flip(w, [1, 2, 3])
        w14 = torch.flip(w, [0, 1, 2, 3])

        return x0, y0, y1, y2, z0, z1, z2, z3, z4, z5, z6, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(36)
    y = torch.rand(14, 17)
    z = torch.rand(13, 14, 15)
    w = torch.rand(48, 12, 16, 17)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_flip.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_flip.pt inputshape=[36],[14,17],[13,14,15],[48,12,16,17]")

    # pnnx inference
    import test_torch_flip_pnnx
    b = test_torch_flip_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
