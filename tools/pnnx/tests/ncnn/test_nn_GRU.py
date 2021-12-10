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

        self.gru_0_0 = nn.GRU(input_size=32, hidden_size=16)
        self.gru_0_1 = nn.GRU(input_size=16, hidden_size=16, num_layers=3, bias=False)
        self.gru_0_2 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, bidirectional=True)
        self.gru_0_3 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, bidirectional=True)
        self.gru_0_4 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, bidirectional=True)

        self.gru_1_0 = nn.GRU(input_size=25, hidden_size=16, batch_first=True)
        self.gru_1_1 = nn.GRU(input_size=16, hidden_size=16, num_layers=3, bias=False, batch_first=True)
        self.gru_1_2 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)
        self.gru_1_3 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)
        self.gru_1_4 = nn.GRU(input_size=16, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)

    def forward(self, x, y):
        x = x.permute(1, 0, 2)

        x0, _ = self.gru_0_0(x)
        x1, _ = self.gru_0_1(x0)
        x2, h0 = self.gru_0_2(x1)
        x3, h1 = self.gru_0_3(x1, h0)
        x4, _ = self.gru_0_4(x1, h1)

        y0, _ = self.gru_1_0(y)
        y1, _ = self.gru_1_1(y0)
        y2, h2 = self.gru_1_2(y1)
        y3, h3 = self.gru_1_3(y1, h2)
        y4, _ = self.gru_1_4(y1, h3)

        x2 = x2.permute(1, 0, 2)
        x3 = x3.permute(1, 0, 2)
        x4 = x4.permute(1, 0, 2)

        return x2, x3, x4, y2, y3, y4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 10, 32)
    y = torch.rand(1, 12, 25)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_GRU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_GRU.pt inputshape=[1,10,32],[1,12,25]")

    # ncnn inference
    import test_nn_GRU_ncnn
    b = test_nn_GRU_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            print(a0.shape)
            print(b0.shape)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
