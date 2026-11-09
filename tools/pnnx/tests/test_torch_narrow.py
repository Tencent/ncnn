# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.narrow(x, 0, 0, 2)
        out1 = torch.narrow(x, 1, 1, 2)
        out2 = torch.narrow(y, 0, 0, 2)
        out3 = torch.narrow(y, 1, 1, 2)
        out4 = torch.narrow(z, 0, 0, 2)
        out5 = torch.narrow(z, 1, 1, 2)
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 3)
    y = torch.rand(5, 3)
    z = torch.rand(3, 5)
    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_narrow.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_narrow.pt inputshape=[3,3],[5,3],[3,5]")

    # pnnx inference
    import test_torch_narrow_pnnx
    b = test_torch_narrow_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_torch_narrow_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_narrow_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_narrow_export_pnnx
    b = test_torch_narrow_export_pnnx.test_inference()

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
