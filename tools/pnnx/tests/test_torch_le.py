# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.le(x, y)
        out1 = torch.le(y, y)
        out2 = torch.le(z, 1)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a0, a1, a2 = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_le",
        pnnx_args=("inputshape=[3,16],[3,16],[5,9,3]",),
    )

    b0, b1, b2 = mod.test_inference()

    passed = torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
