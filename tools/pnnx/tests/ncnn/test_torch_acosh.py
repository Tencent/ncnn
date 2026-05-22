# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.acosh(x + 1)
        y = torch.acosh(y + 1)
        z = torch.acosh(z + 1)
        return x, y, z


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_acosh.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_acosh.pt inputshape=[3,16],[5,9,11],[8,5,9,10]")

    # ncnn inference
    import test_torch_acosh_ncnn
    b = test_torch_acosh_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
