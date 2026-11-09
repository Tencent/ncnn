# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveAvgPool2d(output_size=(7,6))
        self.pool_1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool_2 = nn.AdaptiveAvgPool2d(output_size=(None,3))
        self.pool_3 = nn.AdaptiveAvgPool2d(output_size=(5,None))

    def forward(self, x):
        out0 = self.pool_0(x)
        out1 = self.pool_1(x)
        out2 = self.pool_2(x)
        out3 = self.pool_3(x)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13, 13)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AdaptiveAvgPool2d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AdaptiveAvgPool2d.pt inputshape=[1,128,13,13]")

    # pnnx inference
    import test_nn_AdaptiveAvgPool2d_pnnx
    b = test_nn_AdaptiveAvgPool2d_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_nn_AdaptiveAvgPool2d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_AdaptiveAvgPool2d_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_AdaptiveAvgPool2d_export_pnnx
    c = test_nn_AdaptiveAvgPool2d_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
