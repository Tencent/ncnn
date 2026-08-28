# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

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

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_where",
        pnnx_args=("inputshape=[3,16],[3,16]",),
    )

    b0, b1 = mod.test_inference()

    passed = torch.equal(a0, b0) and torch.equal(a1, b1)
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
