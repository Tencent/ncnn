# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.affine_grid(x, torch.Size((32, 3, 24, 24)), align_corners=False)

        y = F.affine_grid(y, torch.Size((12, 3, 10, 20, 30)), align_corners=False)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(32, 2, 3)
    y = torch.rand(12, 3, 4)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_affine_grid.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_affine_grid.pt inputshape=[32,2,3],[12,3,4]")

    # pnnx inference
    import test_F_affine_grid_pnnx
    b0, b1 = test_F_affine_grid_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
