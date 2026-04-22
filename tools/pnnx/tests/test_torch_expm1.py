# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.expm1(x - 0.5)
        y = torch.expm1(y - 0.5)
        z = torch.expm1(z - 0.5)
        return x, y, z


def test():
    if version.parse(torch.__version__) < version.parse('1.8'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_expm1.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_expm1.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_torch_expm1_pnnx
    b = test_torch_expm1_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
