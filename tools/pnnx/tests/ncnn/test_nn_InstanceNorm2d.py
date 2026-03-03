# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.in_0 = nn.InstanceNorm2d(num_features=12, affine=True)
        self.in_0.weight = nn.Parameter(torch.rand(12))
        self.in_0.bias = nn.Parameter(torch.rand(12))
        self.in_1 = nn.InstanceNorm2d(num_features=12, eps=1e-2, affine=False)

    def forward(self, x):
        x = self.in_0(x)
        x = self.in_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_InstanceNorm2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_InstanceNorm2d.pt inputshape=[1,12,24,64]")

    # ncnn inference
    import test_nn_InstanceNorm2d_ncnn
    b = test_nn_InstanceNorm2d_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
