# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.prelu_0 = nn.PReLU(num_parameters=12)
        self.prelu_1 = nn.PReLU(num_parameters=1, init=0.12)

    def forward(self, x, y, z):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1

        x = self.prelu_0(x)
        x = self.prelu_1(x)

        y = self.prelu_0(y)
        y = self.prelu_1(y)

        z = self.prelu_0(z)
        z = self.prelu_1(z)

        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)
    # w = torch.rand(1, 12, 24, 32, 64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_PReLU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_PReLU.pt inputshape=[1,12],[1,12,64],[1,12,24,64]")

    # ncnn inference
    import test_nn_PReLU_ncnn
    b = test_nn_PReLU_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
