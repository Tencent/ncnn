# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

    pnnx.export(net, "test_F_relu_export", (x, y, z, w))

    # import sys
    # import os
    # sys.path.append(os.path.join(os.getcwd()))

    # fix aten::
    import re
    f=open('test_F_relu_export_pnnx.py','r')
    alllines=f.readlines()
    f.close()
    f=open('test_F_relu_export_pnnx.py','w+')
    for eachline in alllines:
        a=re.sub('aten::','F.',eachline)
        a=re.sub(r'\\', r'\\\\',a)
        f.writelines(a)
    f.close()

    import test_F_relu_export_pnnx
    b0, b1, b2, b3 = test_F_relu_export_pnnx.test_inference()

    assert torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)