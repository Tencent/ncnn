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
        x = torch.diag(x, -1)
        y = torch.diag(y)
        z = torch.diag(z, 3)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(7)
    y = torch.rand(5, 5)
    z = torch.rand(4, 8)
    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_diag",
        pnnx_args=("inputshape=[7],[5,5],[4,8]",),
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
