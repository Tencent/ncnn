# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x[:2,:].fill_(z[0])
        y[:1,:].fill_(0.22)
        return x + y.fill_(7)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)
    z = torch.rand(1)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_fill.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_fill.pt inputshape=[6,16],[6,16],[1]")

    # pnnx inference
    import test_Tensor_fill_pnnx
    b = test_Tensor_fill_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
