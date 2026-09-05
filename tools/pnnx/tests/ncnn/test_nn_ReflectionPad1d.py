# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pad_0 = nn.ReflectionPad1d(2)
        self.pad_1 = nn.ReflectionPad1d(padding=(3,4))
        self.pad_2 = nn.ReflectionPad1d(padding=(1,0))

    def forward(self, x, y):
        x = self.pad_0(x)
        x = self.pad_1(x)
        x = self.pad_2(x)
        y = self.pad_0(y)
        y = self.pad_1(y)
        y = self.pad_2(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13)
    y = torch.rand(2, 12, 13)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_ReflectionPad1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_ReflectionPad1d.pt inputshape=[1,12,13],[2,12,13]")

    # ncnn inference
    import test_nn_ReflectionPad1d_ncnn
    b = test_nn_ReflectionPad1d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
