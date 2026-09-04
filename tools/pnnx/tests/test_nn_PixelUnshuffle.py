# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.down_0 = nn.PixelUnshuffle(2)
        self.down_1 = nn.PixelUnshuffle(4)

    def forward(self, x, y):
        x = self.down_0(x)
        x = self.down_1(x)

        y = self.down_0(y)
        y = self.down_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)
    y = torch.rand(1, 12, 4, 192, 192)

    a0, a1 = net(x, y)

    return test_model_formats(net, (x, y), (a0, a1), "test_nn_PixelUnshuffle")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
