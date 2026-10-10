# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pnnx

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w = F.relu(w)
        return x, y, z, w

def test_export():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a0, a1, a2, a3 = net(x, y, z, w)

    net_pnnx = pnnx.export(net, "test_F_relu_export", (x, y, z, w))

    b0, b1, b2, b3 = net_pnnx(x, y, z, w)

    assert torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)
