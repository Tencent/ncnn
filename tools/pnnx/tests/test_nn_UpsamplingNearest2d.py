# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.up_0 = nn.UpsamplingNearest2d(size=16)
        self.up_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_2 = nn.UpsamplingNearest2d(size=(20,20))
        self.up_3 = nn.UpsamplingNearest2d(scale_factor=(4,4))
        self.up_4 = nn.UpsamplingNearest2d(size=(16,24))
        self.up_5 = nn.UpsamplingNearest2d(scale_factor=(2,3))

        self.up_w = nn.UpsamplingNearest2d(scale_factor=(2.976744,2.976744))

    def forward(self, x, w):
        x = self.up_0(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.up_4(x)
        x = self.up_5(x)

        w = self.up_w(w)
        return x, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, w)

    return test_model_formats(net, (x, w), a, "test_nn_UpsamplingNearest2d")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
