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

        self.pool_0 = nn.AdaptiveMaxPool1d(output_size=(7), return_indices=True)
        self.pool_1 = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x, indices = self.pool_0(x)
        x = self.pool_1(x)
        return x, indices

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13)

    a0, a1 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AdaptiveMaxPool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AdaptiveMaxPool1d.pt inputshape=[1,128,13]")

    # pnnx inference
    import test_nn_AdaptiveMaxPool1d_pnnx
    b0, b1 = test_nn_AdaptiveMaxPool1d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
