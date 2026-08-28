# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.view_as_complex(x)
        y = torch.view_as_complex(y)
        z = torch.view_as_complex(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 2)
    y = torch.rand(1, 5, 9, 2)
    z = torch.rand(14, 8, 5, 9, 2)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_view_as_complex",
        pnnx_args=("inputshape=[1,3,2],[1,5,9,2],[14,8,5,9,2]",),
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