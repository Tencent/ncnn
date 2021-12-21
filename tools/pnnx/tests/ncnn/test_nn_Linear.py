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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.linear_1 = nn.Linear(in_features=16, out_features=3, bias=True)

    def forward(self, x, y):
        x = self.linear_0(x)
        x = self.linear_1(x)

        y = self.linear_0(y)
        y = self.linear_1(y)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(64)
    y = torch.rand(12, 64)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_Linear.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Linear.pt inputshape=[64],[12,64]")

    # ncnn inference
    import test_nn_Linear_ncnn
    b0, b1 = test_nn_Linear_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
