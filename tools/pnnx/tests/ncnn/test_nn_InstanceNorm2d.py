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

    def forward(self, x, q):
        x = self.in_0(x)
        x = self.in_1(x)
        q = self.in_0(q)
        q = self.in_1(q)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    q = torch.rand(2, 12, 24, 64)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_nn_InstanceNorm2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_InstanceNorm2d.pt inputshape=[1,12,24,64],[2,12,24,64]")

    # ncnn inference
    import test_nn_InstanceNorm2d_ncnn
    b = test_nn_InstanceNorm2d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
