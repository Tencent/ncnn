# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.MaxPool3d(kernel_size=3)
        self.pool_1 = nn.MaxPool3d(kernel_size=4, stride=2, padding=2, dilation=1)
        self.pool_2 = nn.MaxPool3d(kernel_size=(1,2,3), stride=1, padding=(0,0,1), dilation=1, return_indices=False, ceil_mode=False)
        self.pool_3 = nn.MaxPool3d(kernel_size=(3,4,5), stride=(1,2,2), padding=(1,2,2), dilation=1, return_indices=False, ceil_mode=True)
        self.pool_4 = nn.MaxPool3d(kernel_size=(2,3,3), stride=1, padding=1, dilation=(1,2,2), return_indices=False, ceil_mode=False)
        self.pool_5 = nn.MaxPool3d(kernel_size=2, stride=1, padding=0, dilation=1, return_indices=True, ceil_mode=True)
        self.pool_6 = nn.MaxPool3d(kernel_size=(5,4,4), stride=1, padding=2, dilation=1, return_indices=True, ceil_mode=False)

    def forward(self, x, y):
        x = self.pool_0(x)
        x = self.pool_1(x)
        x = self.pool_2(x)
        x = self.pool_3(x)
        x = self.pool_4(x)
        x, tx = self.pool_5(x)
        x, indices = self.pool_6(x)

        y = self.pool_0(y)
        y = self.pool_1(y)
        y = self.pool_2(y)
        y = self.pool_3(y)
        y = self.pool_4(y)
        y, ty = self.pool_5(y)
        return x, indices, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64, 64)
    y = torch.rand(12, 64, 64, 64)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_MaxPool3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_MaxPool3d.pt inputshape=[1,12,64,64,64],[12,64,64,64]")

    # pnnx inference
    import test_nn_MaxPool3d_pnnx
    b = test_nn_MaxPool3d_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_nn_MaxPool3d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_MaxPool3d_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_MaxPool3d_export_pnnx
    c = test_nn_MaxPool3d_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype:
            return False
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
