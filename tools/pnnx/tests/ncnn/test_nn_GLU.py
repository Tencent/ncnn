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

    def forward(self, x, y, z):
        x = self.act_0(x)
        y = self.act_1(y)
        z = self.act_2(z)
        z2 = self.act_3(z)
        return x, y, z, z2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(12, 64)
    z = torch.rand(12, 24, 64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_GLU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_GLU.pt inputshape=[12],[12,64],[12,24,64]")

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
