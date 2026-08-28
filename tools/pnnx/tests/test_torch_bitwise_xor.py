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
        a = torch.ge(x, y)
        b = torch.ne(x, y)
        out = torch.bitwise_xor(a, b)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_bitwise_xor",
        pnnx_args=("inputshape=[3,16],[3,16]",),
    )

    b = mod.test_inference()

    passed = torch.equal(a, b)
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
