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

        self.prelu_0 = nn.PReLU(num_parameters=12)
        self.prelu_1 = nn.PReLU(num_parameters=1, init=0.12)

    def forward(self, x, y, z):
        x = self.prelu_0(x)
        x = self.prelu_1(x)

        y = self.prelu_0(y)
        y = self.prelu_1(y)

        z = self.prelu_0(z)
        z = self.prelu_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_PReLU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_PReLU.pt inputshape=[1,12],[1,12,64],[1,12,24,64]")

    # ncnn inference
    import test_nn_PReLU_ncnn
    b = test_nn_PReLU_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
