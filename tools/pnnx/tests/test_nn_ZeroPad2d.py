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

        self.pad_0 = nn.ZeroPad2d(2)
        self.pad_1 = nn.ZeroPad2d(padding=(3,4,5,6))
        self.pad_2 = nn.ZeroPad2d(padding=(1,0,2,0))

    def forward(self, x):
        x = self.pad_0(x)
        x = self.pad_1(x)
        x = self.pad_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13, 13)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_ZeroPad2d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_ZeroPad2d.pt inputshape=[1,12,13,13]")

    # pnnx inference
    import test_nn_ZeroPad2d_pnnx
    b = test_nn_ZeroPad2d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
