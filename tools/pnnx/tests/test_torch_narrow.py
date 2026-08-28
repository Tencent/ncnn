# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

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

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_narrow",
        pnnx_args=("inputshape=[3,3],[5,3],[3,5]",),
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
