# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from pnnx_test_utils import convert_and_import


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.log1p(x)
        y = torch.log1p(y)
        z = torch.log1p(z)
        return x, y, z


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_log1p",
        pnnx_args=("inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]",),
    )

    b = mod.test_inference()

    passed = True
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-6, 1e-6):
            passed = False
    return passed


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
