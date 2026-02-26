# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = F.normalize(x)
        x = F.normalize(x, eps=1e-3)

        y = F.normalize(y, p=1, dim=1)
        y = F.normalize(y, dim=2)

        z = F.normalize(z)
        z = F.normalize(z, dim=2, eps=1e-4)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_normalize.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_normalize.pt inputshape=[1,24,64],[1,12,24,64],[1,12,16,24,64]")

    # pnnx inference
    import test_F_normalize_pnnx
    b0, b1, b2 = test_F_normalize_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
