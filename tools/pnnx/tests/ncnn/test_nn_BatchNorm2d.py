# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bn_0 = nn.BatchNorm2d(num_features=32)
        self.bn_1 = nn.BatchNorm2d(num_features=32, eps=1e-1, affine=False)
        self.bn_2 = nn.BatchNorm2d(num_features=11, affine=True)

    def forward(self, x, y, q):
        x = self.bn_0(x)
        x = self.bn_1(x)

        y = self.bn_2(y)

        q = self.bn_0(q)
        q = self.bn_1(q)

        return x, y, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 32, 12, 64)
    y = torch.rand(1, 11, 1, 1)
    q = torch.rand(2, 32, 12, 64)

    a0, a1, a2 = net(x, y, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, q))
    mod.save("test_nn_BatchNorm2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_BatchNorm2d.pt inputshape=[1,32,12,64],[1,11,1,1],[2,32,12,64]")

    # ncnn inference
    import test_nn_BatchNorm2d_ncnn
    b0, b1, b2 = test_nn_BatchNorm2d_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4) and torch.allclose(a2, b2, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
