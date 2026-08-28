# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y0, y1, z):
        out0 = torch.scatter_add(x, 0, y0, z)
        out1 = torch.scatter_add(x, 0, y1, z)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(13, 15)
    y0 = torch.randint(10, (1, 15), dtype=torch.long)
    y1 = torch.randint(10, (12, 15), dtype=torch.long)
    z = torch.rand(12, 15)

    a = net(x, y0, y1, z)

    mod = convert_and_import(
        net,
        (x, y0, y1, z),
        "test_torch_scatter_add",
        pnnx_args=("inputshape=[13,15],[1,15]i64,[12,15]i64,[12,15]",),
    )

    b = mod.test_inference()

    passed = True
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            passed = False
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
