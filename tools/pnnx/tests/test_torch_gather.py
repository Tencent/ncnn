# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = torch.gather(x, 0, y)
        out1 = torch.gather(x, 1, y)
        out2 = torch.gather(x, 2, y)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(10, 13, 16)
    y = torch.randint(10, (10, 13, 16), dtype=torch.long)

    a = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_gather",
        pnnx_args=("inputshape=[10,13,16],[10,13,16]i64",),
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
