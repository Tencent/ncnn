# Copyright 2026 ktz03
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool1d = nn.AvgPool1d(2, stride=2, ceil_mode=True)
        self.pool1d_pad = nn.AvgPool1d(3, stride=2, padding=1, ceil_mode=True)
        self.pool2d = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        self.pool2d_pad = nn.AvgPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.pool3d = nn.AvgPool3d(2, stride=2, ceil_mode=True)
        self.pool3d_pad = nn.AvgPool3d(3, stride=2, padding=1, ceil_mode=True)

    def forward(self, x, xp, y, yp, z, zp):
        return (F.avg_pool1d(x, 2, stride=2, ceil_mode=True, count_include_pad=False),
                F.avg_pool2d(y, 2, stride=2, ceil_mode=False),
                F.avg_pool1d(x, 2, stride=2, ceil_mode=True),
                F.avg_pool1d(xp, 3, stride=2, padding=1, ceil_mode=True),
                self.pool1d(x), self.pool1d_pad(xp),
                F.avg_pool2d(y, 2, stride=2, ceil_mode=True),
                F.avg_pool2d(yp, 3, stride=2, padding=1, ceil_mode=True),
                self.pool2d(y), self.pool2d_pad(yp),
                F.avg_pool3d(z, 2, stride=2, ceil_mode=True),
                F.avg_pool3d(zp, 3, stride=2, padding=1, ceil_mode=True),
                self.pool3d(z), self.pool3d_pad(zp))

def test():
    net = Model().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 2, 3)
    xp = torch.rand(1, 2, 4)
    y = torch.rand(1, 2, 3, 5)
    yp = torch.rand(1, 2, 4, 6)
    z = torch.rand(1, 2, 3, 5, 3)
    zp = torch.rand(1, 2, 4, 6, 4)
    inputs = (x, xp, y, yp, z, zp)
    a = net(*inputs)

    mod = torch.jit.trace(net, inputs)
    mod.save("test_avg_pool_ceil_mode.pt")

    import os
    os.system("../../src/pnnx test_avg_pool_ceil_mode.pt inputshape=[1,2,3],[1,2,4],[1,2,3,5],[1,2,4,6],[1,2,3,5,3],[1,2,4,6,4] fp16=0")

    import test_avg_pool_ceil_mode_ncnn
    b = test_avg_pool_ceil_mode_ncnn.test_inference()
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
