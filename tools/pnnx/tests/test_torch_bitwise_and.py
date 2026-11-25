# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        a = torch.ge(x, y)
        b = torch.ne(x, y)
        out = torch.bitwise_and(a, b)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_bitwise_and.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_bitwise_and.pt inputshape=[3,16],[3,16]")

    # pnnx inference
    import test_torch_bitwise_and_pnnx
    b = test_torch_bitwise_and_pnnx.test_inference()

    if not (torch.equal(a, b)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_torch_bitwise_and_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_bitwise_and_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_bitwise_and_export_pnnx
    c = test_torch_bitwise_and_export_pnnx.test_inference()

    return a.dtype == c.dtype and torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
