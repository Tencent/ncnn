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

        self.w2 = nn.Parameter(torch.rand(6, 12, 4, 4))
        self.b2 = nn.Parameter(torch.rand(12))
        self.w3 = nn.Parameter(torch.rand(12, 2, 3, 3))

    def forward(self, x, w0, w1, b1, y):
        x = F.conv_transpose2d(x, w0, None, stride=(2,2), padding=(1,1), output_padding=(1,1))
        x = F.conv_transpose2d(x, w1, b1, stride=(1,2), padding=(2,1), dilation=(2,1), groups=2)

        y = F.conv_transpose2d(y, self.w2, self.b2, stride=(2,2), padding=(1,1), output_padding=(1,1))
        y = F.conv_transpose2d(y, self.w3, None, stride=(1,2), padding=(2,1), dilation=(2,1), groups=3)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 22, 32)
    w0 = torch.rand(12, 16, 3, 3)
    w1 = torch.rand(16, 8, 5, 5)
    b1 = torch.rand(16)
    y = torch.rand(1, 6, 5, 6)

    a0, a1 = net(x, w0, w1, b1, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, w0, w1, b1, y))
    mod.save("test_F_conv_transpose2d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_conv_transpose2d.pt inputshape=[1,12,22,32],[12,16,3,3],[16,8,5,5],[16],[1,6,5,6]")

    # pnnx inference
    import test_F_conv_transpose2d_pnnx
    b0, b1 = test_F_conv_transpose2d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
