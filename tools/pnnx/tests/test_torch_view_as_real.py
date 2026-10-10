# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.view_as_real(x)
        y = torch.view_as_real(y)
        z = torch.view_as_real(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16,dtype=torch.complex64)
    y = torch.rand(1, 5, 9, 11,dtype=torch.complex64)
    z = torch.rand(14, 8, 5, 9, 10,dtype=torch.complex64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_view_as_real.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_view_as_real.pt inputshape=[1,3,16]c64,[1,5,9,11]c64,[14,8,5,9,10]c64")

    # pnnx inference
    import test_torch_view_as_real_pnnx
    b = test_torch_view_as_real_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)