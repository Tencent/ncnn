# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.down_0 = nn.PixelUnshuffle(2)
        self.down_1 = nn.PixelUnshuffle(4)

    def forward(self, x, y):
        x = self.down_0(x)
        x = self.down_1(x)

        # onnx export only supports 4d
        # y = self.down_0(y)
        # y = self.down_1(y)
        y = y.relu()
        return x, y

def test():
    if version.parse(torch.__version__) < version.parse('1.12'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)
    y = torch.rand(1, 12, 4, 192, 192)

    a0, a1 = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_nn_PixelUnshuffle.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_PixelUnshuffle.onnx inputshape=[1,3,128,128],[1,12,4,192,192]")

    # pnnx inference
    import test_nn_PixelUnshuffle_pnnx
    b0, b1 = test_nn_PixelUnshuffle_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
