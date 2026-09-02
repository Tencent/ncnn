# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.bn_0 = nn.BatchNorm1d(num_features=16)
        self.linear_1 = nn.Linear(in_features=16, out_features=3, bias=True)
        self.bn_1 = nn.BatchNorm1d(num_features=3)

    def forward(self, x, y):
        x = self.linear_0(x)
        x = self.bn_0(x)
        x = self.linear_1(x)
        x = self.bn_1(x)

        y = self.linear_0(y)
        y = self.bn_0(y)
        y = self.linear_1(y)
        y = self.bn_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64)
    y = torch.rand(12, 64)

    a0, a1 = net(x, y)

    return test_model_formats(
        net,
        (x, y),
        (a0, a1),
        "test_pnnx_fuse_linear_batchnorm1d",
        compare=lambda a, b: torch.allclose(a, b, 1e-4, 1e-4),
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
