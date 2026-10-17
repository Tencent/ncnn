# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, torchscript_to_pnnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w0, w1, b1):
        x = F.linear(x, w0, None)
        x = F.linear(x, w1, b1)

        y = F.linear(y, w0, None)
        y = F.linear(y, w1, b1)

        z = F.linear(z, w0, None)
        z = F.linear(z, w1, b1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w0 = torch.rand(12, 16)
    w1 = torch.rand(32, 12)
    bias1 = torch.rand(32)

    a0, a1, a2 = net(x, y, z, w0, w1, bias1)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w0, w1, bias1))
    mod.save("test_F_linear.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_F_linear", "[1,16],[12,2,16],[1,3,12,16],[12,16],[32,12],[32]")

    # pnnx inference
    b0, b1, b2 = converted(x, y, z, w0, w1, bias1)

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)):
        return False

    if not has_torch_export():
        return True

    converted = exported_program_to_pnnx(net, (x, y, z, w0, w1, bias1), "test_F_linear_pt2")
    c0, c1, c2 = converted(x, y, z, w0, w1, bias1)

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
