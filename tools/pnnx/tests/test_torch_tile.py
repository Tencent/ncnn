# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.tile(x, (1, 2, 3))
        x = torch.tile(x, (2, 3, 4))
        y = torch.tile(y, (1, 2, 1, 4))
        y = torch.tile(y, (3, 4, 5, 1))
        z = torch.tile(z, (1, 2, 3, 1, 5))
        z = torch.tile(z, (2, 3, 3, 1, 1))
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
    mod.save("test_torch_tile.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_tile.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_torch_tile_pnnx
    b = test_torch_tile_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_torch_tile_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_tile_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_tile_export_pnnx
    b = test_torch_tile_export_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape or a0.dtype != b0.dtype or not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
