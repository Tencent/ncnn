# Copyright 2021 Tencent
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

    def forward(self, x):
        x = F.relu(x)
        return x

def test_export():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    x2 = torch.rand(1, 128)

    a0 = net(x)
    a1 = net(x2)

    net_pnnx = pnnx.export(net, "test_F_relu_dexport", x, x2)

    b0 = net_pnnx(x)
    b1 = net_pnnx(x2)

    assert torch.equal(a0, b0) and torch.equal(a1, b1)
