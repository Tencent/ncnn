# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.logaddexp(x, y)
        out1 = torch.logaddexp(y, y)
        out2 = torch.logaddexp(z, torch.ones_like(z) + 0.5)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_logaddexp",
        pnnx_args=("inputshape=[3,16],[3,16],[5,9,3]",),
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
