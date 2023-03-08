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

        self.shuffle_0 = nn.ChannelShuffle(2)
        self.shuffle_1 = nn.ChannelShuffle(16)

    def forward(self, x, y):
        x = self.shuffle_0(x)
        x = self.shuffle_1(x)

        y = self.shuffle_0(y)
        y = self.shuffle_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64, 6, 8)
    y = torch.rand(1, 96, 7, 9)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_ChannelShuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_ChannelShuffle.pt inputshape=[1,64,6,8],[1,96,7,9]")

    # pnnx inference
    import test_nn_ChannelShuffle_pnnx
    b0, b1 = test_nn_ChannelShuffle_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
