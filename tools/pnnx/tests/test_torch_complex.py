# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.complex(x, y)
        z = torch.complex(z, w)
        return x, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 3, 16)
    z = torch.rand(14, 5, 9, 10)
    w = torch.rand(14, 5, 9, 10)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_complex.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_complex.pt inputshape=[1,3,16],[1,3,16],[14,5,9,10],[14,5,9,10]")

    # pnnx inference
    import test_torch_complex_pnnx
    b = test_torch_complex_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z, w))
    torch.export.save(ep, "test_torch_complex_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_complex_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_complex_export_pnnx
    c = test_torch_complex_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype or not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
