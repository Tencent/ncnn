# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.permute(1, 0, 2).transpose(2, 1)
        y = y.transpose(1, -1).permute(3, 2, 1, 0)
        z = z.permute(1, 2, 3, 4, 0).transpose(0, 1).transpose(-3, 4).permute(4, 3, 2, 1, 0).permute(3, 1, 0, 4, 2)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(8, 9, 10)
    y = torch.rand(9, 10, 11, 12)
    z = torch.rand(1, 9, 10, 11, 12)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_pnnx_fuse_adjacent_permute",
        pnnx_args=("inputshape=[8,9,10],[9,10,11,12],[1,9,10,11,12]",),
    )
    b = mod.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
