# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        c0 = torch.le(x, y)
        c1 = torch.ge(y, y)
        out0 = torch.where(c0, x + 2, x + 4)
        out1 = torch.where(c1, x + y, x - y)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_where.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_where.pt inputshape=[3,16],[3,16]")

    # pnnx inference
    import test_torch_where_pnnx
    b0, b1 = test_torch_where_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
