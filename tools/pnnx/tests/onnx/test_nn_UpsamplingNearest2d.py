# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        if version.parse(torch.__version__) < version.parse('1.12'):
            self.up_0 = nn.UpsamplingNearest2d(size=60)
            self.up_1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.up_2 = nn.UpsamplingNearest2d(size=(40,40))
            self.up_3 = nn.UpsamplingNearest2d(scale_factor=(4,4))
            self.up_4 = nn.UpsamplingNearest2d(size=(60,40))
            self.up_5 = nn.UpsamplingNearest2d(scale_factor=(2,3))
        else:
            self.up_0 = nn.UpsamplingNearest2d(size=16)
            self.up_1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.up_2 = nn.UpsamplingNearest2d(size=(20,20))
            self.up_3 = nn.UpsamplingNearest2d(scale_factor=(4,4))
            self.up_4 = nn.UpsamplingNearest2d(size=(16,24))
            self.up_5 = nn.UpsamplingNearest2d(scale_factor=(2,3))

        self.up_w = nn.UpsamplingNearest2d(scale_factor=(2.976744,2.976744))

    def forward(self, x, w):
        if version.parse(torch.__version__) < version.parse('1.12'):
            x0 = self.up_0(x)
            x0 = self.up_1(x0)
            x1 = self.up_2(x)
            x1 = self.up_3(x1)
            x2 = self.up_4(x)
            x2 = self.up_5(x2)

            w = self.up_w(w)
            return x0, x1, x2, w
        else:
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

    # export onnx
    torch.onnx.export(net, (x, w), "test_nn_UpsamplingNearest2d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_UpsamplingNearest2d.onnx inputshape=[1,3,32,32],[1,8,86,86]")

    # pnnx inference
    import test_nn_UpsamplingNearest2d_pnnx
    b = test_nn_UpsamplingNearest2d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
