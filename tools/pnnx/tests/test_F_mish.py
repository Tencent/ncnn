# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

def mish_forward_0(x):
    return x * F.softplus(x).tanh()

def mish_forward_1(x):
    return x.mul(torch.tanh(F.softplus(x)))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1
        x = F.mish(x)
        y = F.mish(y)
        z = mish_forward_0(z)
        w = mish_forward_1(w)
        return x, y, z, w

def test():
    if version.parse(torch.__version__) < version.parse('1.9'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_F_mish.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_mish.pt inputshape=[1,16],[12,2,16],[1,3,12,16],[1,5,7,9,11]")

    # pnnx inference
    import test_F_mish_pnnx
    b = test_F_mish_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
