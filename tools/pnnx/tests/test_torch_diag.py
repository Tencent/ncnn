# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.diag(x, -1)
        y = torch.diag(y)
        z = torch.diag(z, 3)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(7)
    y = torch.rand(5, 5)
    z = torch.rand(4, 8)
    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_diag.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_diag.pt inputshape=[7],[5,5],[4,8]")

    # pnnx inference
    import test_torch_diag_pnnx
    b = test_torch_diag_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
