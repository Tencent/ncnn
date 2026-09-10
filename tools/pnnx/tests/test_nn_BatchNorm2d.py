# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bn_0 = nn.BatchNorm2d(num_features=32)
        self.bn_1 = nn.BatchNorm2d(num_features=32, eps=1e-1, affine=False)
        self.bn_2 = nn.BatchNorm2d(num_features=11, affine=True)

    def forward(self, x, y):
        x = self.bn_0(x)
        x = self.bn_1(x)

        y = self.bn_2(y)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 32, 12, 64)
    y = torch.rand(1, 11, 1, 1)

    a0, a1 = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_nn_BatchNorm2d",
        pnnx_args=("inputshape=[1,32,12,64],[1,11,1,1]",),
    )

    b0, b1 = mod.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
