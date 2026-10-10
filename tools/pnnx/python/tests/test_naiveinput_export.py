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

    a0 = net(x)

    net_pnnx = pnnx.export(net, "test_F_relu_nexport", x)

    b0 = net_pnnx(x)

    assert torch.equal(a0, b0)
