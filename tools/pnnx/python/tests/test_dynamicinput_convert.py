# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

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
