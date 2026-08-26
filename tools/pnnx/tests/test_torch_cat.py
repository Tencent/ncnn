# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, torchscript_to_pnnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.cat((x, y), dim=1)
        out1 = torch.cat((z, w), dim=3)
        out2 = torch.cat((w, w), dim=2)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 2, 16)
    z = torch.rand(1, 5, 9, 11)
    w = torch.rand(1, 5, 9, 3)

    a0, a1, a2 = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_cat.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_torch_cat", "[1,3,16],[1,2,16],[1,5,9,11],[1,5,9,3]")

    # pnnx inference
    b0, b1, b2 = converted(x, y, z, w)

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)):
        return False

    if not has_torch_export():
        return True

    converted = exported_program_to_pnnx(net, (x, y, z, w), "test_torch_cat_pt2")
    c0, c1, c2 = converted(x, y, z, w)

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
