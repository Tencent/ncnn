# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm_0_0 = nn.LSTM(input_size=32, hidden_size=16)
        self.lstm_0_1 = nn.LSTM(input_size=16, hidden_size=16, num_layers=3, bias=False)
        self.lstm_0_2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=4, bias=True, bidirectional=True)
        self.lstm_0_3 = nn.LSTM(input_size=32, hidden_size=16, num_layers=4, bias=True, bidirectional=True)

        self.lstm_1_0 = nn.LSTM(input_size=25, hidden_size=16, batch_first=True)
        self.lstm_1_1 = nn.LSTM(input_size=16, hidden_size=16, num_layers=3, bias=False, batch_first=True)
        self.lstm_1_2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)
        self.lstm_1_3 = nn.LSTM(input_size=32, hidden_size=16, num_layers=4, bias=True, batch_first=True, bidirectional=True)

    def forward(self, x, y):
        x0, (h0, c0) = self.lstm_0_0(x)
        x1, _ = self.lstm_0_1(x0)
        x2, (h2, c2) = self.lstm_0_2(x1)
        x3, (h3, c3) = self.lstm_0_3(x2, (h2, c2))

        y0, (h4, c4) = self.lstm_1_0(y)
        y1, _ = self.lstm_1_1(y0)
        y2, (h6, c6) = self.lstm_1_2(y1)
        y3, (h7, c7) = self.lstm_1_3(y2, (h6, c6))
        return x2, x3, h0, h2, h3, c0, c2, c3, y2, y3, h4, h6, h7, c4, c6, c7

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(10, 1, 32)
    y = torch.rand(1, 12, 25)

    a = net(x, y)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x, y), "test_nn_LSTM.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x, y), "test_nn_LSTM.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_LSTM.onnx inputshape=[10,1,32],[1,12,25]")

    # pnnx inference
    import test_nn_LSTM_pnnx
    b = test_nn_LSTM_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
