# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out = torch.mv(x, y)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3)
    y = torch.rand(3)

    a = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_mv",
        pnnx_args=("inputshape=[2,3],[3]",),
    )

    b = mod.test_inference()

    passed = torch.equal(a, b)
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
