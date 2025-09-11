# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.cross(x, y)
        out1 = torch.cross(x, y, dim=1)
        out2 = torch.cross(z, w)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 3)
    y = torch.rand(3, 3)
    z = torch.rand(5, 3)
    w = torch.rand(5, 3)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_cross.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_cross.pt inputshape=[3,3],[3,3],[5,3],[5,3]")

    # pnnx inference
    import test_torch_cross_pnnx
    b = test_torch_cross_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
