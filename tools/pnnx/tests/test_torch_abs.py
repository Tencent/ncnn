# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.abs(x - 0.5)
        y = torch.abs(y - 0.5)
        z = torch.abs(z - 0.5)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_abs.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_abs.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_torch_abs_pnnx
    b = test_torch_abs_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_torch_abs_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_abs_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_abs_export_pnnx
    c = test_torch_abs_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
