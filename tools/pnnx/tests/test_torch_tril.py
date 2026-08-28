# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from pnnx_test_utils import convert_and_import


class Model(nn.Module):
    def forward(self, x, y, z):
        x = torch.tril(x, diagonal=-2)
        y = torch.tril(y)
        z = torch.tril(z, diagonal=2)
        return x, y, z


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(7, 9)
    y = torch.rand(3, 5, 5)
    z = torch.rand(2, 4, 6, 8)
    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_tril",
        pnnx_args=("inputshape=[7,9],[3,5,5],[2,4,6,8]",),
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
