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
        out0 = torch.eq(x, y)
        out1 = torch.eq(y, y)
        out2 = torch.eq(z, 3)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_eq.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_eq.pt inputshape=[3,16],[3,16],[5,9,3]")

    # pnnx inference
    import test_torch_eq_pnnx
    b0, b1, b2 = test_torch_eq_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_torch_eq_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_eq_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_eq_export_pnnx
    c0, c1, c2 = test_torch_eq_export_pnnx.test_inference()

    return a0.dtype == c0.dtype and a1.dtype == c1.dtype and a2.dtype == c2.dtype and torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
