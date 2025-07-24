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
    x1 = torch.rand(1, 128)

    a0 = net(x)
    a1 = net(x1)

    mod = torch.jit.trace(net, x)
    mod.save("test_F_relu_dconvert.pt")

    net2 = pnnx.convert("test_F_relu_dconvert.pt", x, x1)

    b0 = net2(x)
    b1 = net2(x1)

    assert torch.equal(a0, b0) and torch.equal(a1, b1)
