# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bn_0 = nn.BatchNorm1d(num_features=32)
        self.bn_1 = nn.BatchNorm1d(num_features=32, eps=1e-1, affine=False)
        self.bn_2 = nn.BatchNorm1d(num_features=11, affine=True)

    def forward(self, x, y):
        x = self.bn_0(x)
        x = self.bn_1(x)

        y = self.bn_2(y)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 32, 64)
    y = torch.rand(1, 11, 1)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_BatchNorm1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_BatchNorm1d.pt inputshape=[1,32,64],[1,11,1]")

    # ncnn inference
    import test_nn_BatchNorm1d_ncnn
    b0, b1 = test_nn_BatchNorm1d_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
