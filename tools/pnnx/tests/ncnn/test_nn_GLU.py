# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.act_0 = nn.GLU(dim=0)
        self.act_1 = nn.GLU(dim=1)
        self.act_2 = nn.GLU(dim=2)
        self.act_3 = nn.GLU(dim=-1)
        self.act_4 = nn.GLU(dim=3)

    def forward(self, x, y, z, w):
        x = self.act_0(x)
        y = self.act_1(y)
        z = self.act_2(z)
        z2 = self.act_3(z)
        w0 = self.act_0(w)
        w1 = self.act_1(w)
        w2 = self.act_2(w)
        w3 = self.act_4(w)
        w4 = self.act_3(w)
        return x, y, z, z2, w0, w1, w2, w3, w4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(12, 64)
    z = torch.rand(12, 24, 64)
    w = torch.rand(8, 10, 12, 14)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_nn_GLU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_GLU.pt inputshape=[12],[12,64],[12,24,64],[8,10,12,14]")

    # ncnn inference
    import test_nn_GLU_ncnn
    b = test_nn_GLU_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
