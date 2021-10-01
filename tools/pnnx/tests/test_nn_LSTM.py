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

        self.lstm_0_0 = nn.LSTM(input_size=32, hidden_size=16)
        self.lstm_0_1 = nn.LSTM(input_size=16, hidden_size=16, num_layers=3, bias=False)
        self.lstm_0_2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=4, bias=True, bidirectional=True)

        self.lstm_1_0 = nn.LSTM(input_size=25, hidden_size=16, batch_first=True)
        self.lstm_1_1 = nn.LSTM(input_size=16, hidden_size=16, num_layers=3, bias=False, batch_first=True)
        self.lstm_1_2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)

    def forward(self, x, y):
        x, (h0, c0) = self.lstm_0_0(x)
        x, (h1, c1) = self.lstm_0_1(x)
        x, (h2, c2) = self.lstm_0_2(x)

        y, (h3, c3) = self.lstm_1_0(y)
        y, (h4, c4) = self.lstm_1_1(y)
        y, (h5, c5) = self.lstm_1_2(y)
        return x, h0, h1, h2, c0, c1, c2, y, h3, h4, h5, c3, c4, c5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(10, 1, 32)
    y = torch.rand(1, 12, 25)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_LSTM.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_LSTM.pt inputshape=[10,1,32],[1,12,25]")

    # pnnx inference
    import test_nn_LSTM_pnnx
    b = test_nn_LSTM_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
