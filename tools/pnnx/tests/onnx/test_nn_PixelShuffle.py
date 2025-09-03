# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.up_0 = nn.PixelShuffle(4)
        self.up_1 = nn.PixelShuffle(2)

    def forward(self, x, y):
        x = self.up_0(x)
        x = self.up_1(x)

        # onnx export only supports 4d
        # y = self.up_0(y)
        # y = self.up_1(y)
        y = y.relu()
        return x, y

def test():
    if version.parse(torch.__version__) < version.parse('1.12'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 6, 8)
    y = torch.rand(1, 12, 192, 7, 9)

    a0, a1 = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_nn_PixelShuffle.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_PixelShuffle.onnx inputshape=[1,128,6,8],[1,12,192,7,9]")

    # pnnx inference
    import test_nn_PixelShuffle_pnnx
    b0, b1 = test_nn_PixelShuffle_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
