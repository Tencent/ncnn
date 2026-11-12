# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # ceil_mode pads the last window with zeros,
        # torch still divides by the number of real elements
        self.pool_0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=True)
        self.pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True, count_include_pad=True)

    def forward(self, x):
        a = self.pool_0(x)
        b = self.pool_1(x)
        return a, b

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, 7, 9)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AvgPool2d_ceil.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AvgPool2d_ceil.pt inputshape=[1,1,7,9]")

    # pnnx inference
    import test_nn_AvgPool2d_ceil_pnnx
    b = test_nn_AvgPool2d_ceil_pnnx.test_inference()

    if len(a) != len(b):
        return False

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0):
            return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
