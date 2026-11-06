# Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0 = F.glu(x, dim=0)

        y0 = F.glu(y, dim=0)
        y1 = F.glu(y, dim=1)

        z0 = F.glu(z, dim=0)
        z1 = F.glu(z, dim=1)
        z2 = F.glu(z, dim=2)
        return x0, y0, y1, z0, z1, z2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18)
    y = torch.rand(12, 16)
    z = torch.rand(24, 28, 34)

    x0, y0, y1, z0, z1, z2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_glu.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_glu.pt inputshape=[18],[12,16],[24,28,34]")

    # pnnx inference
    import test_F_glu_pnnx
    x0p, y0p, y1p, z0p, z1p, z2p = test_F_glu_pnnx.test_inference()

    if not (torch.equal(x0, x0p) and torch.equal(y0, y0p) and torch.equal(y1, y1p) and torch.equal(z0, z0p) and torch.equal(z1, z1p) and torch.equal(z2, z2p)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_F_glu_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_glu_export.pt2 fp16=0")

    # pnnx inference
    import test_F_glu_export_pnnx
    x0p_export, y0p_export, y1p_export, z0p_export, z1p_export, z2p_export = test_F_glu_export_pnnx.test_inference()

    return torch.equal(x0, x0p_export) and torch.equal(y0, y0p_export) and torch.equal(y1, y1p_export) \
            and torch.equal(z0, z0p_export) and torch.equal(z1, z1p_export) and torch.equal(z2, z2p_export)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
