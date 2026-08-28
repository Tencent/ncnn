# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out = torch.bitwise_left_shift(x, y)
        return out

def test():
    if version.parse(torch.__version__) < version.parse('1.10'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (3, 16), dtype=torch.int)
    y = torch.randint(10, (3, 16), dtype=torch.int)

    a = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_bitwise_left_shift",
        pnnx_args=("inputshape=[3,16]i32,[3,16]i32",),
    )

    b = mod.test_inference()

    passed = torch.equal(a, b)
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
