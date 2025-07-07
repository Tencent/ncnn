# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gn_0 = nn.GroupNorm(num_groups=4, num_channels=12)
        self.gn_0.weight = nn.Parameter(torch.rand(12))
        self.gn_0.bias = nn.Parameter(torch.rand(12))
        self.gn_1 = nn.GroupNorm(num_groups=12, num_channels=12, eps=1e-2, affine=False)
        self.gn_2 = nn.GroupNorm(num_groups=1, num_channels=12, eps=1e-4, affine=True)
        self.gn_2.weight = nn.Parameter(torch.rand(12))
        self.gn_2.bias = nn.Parameter(torch.rand(12))

    def forward(self, x, y, z):
        x = self.gn_0(x)
        x = self.gn_1(x)
        x = self.gn_2(x)

        y = self.gn_0(y)
        y = self.gn_1(y)
        y = self.gn_2(y)

        z = self.gn_0(z)
        z = self.gn_1(z)
        z = self.gn_2(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 24, 32, 64)

    a0, a1, a2 = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_nn_GroupNorm.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_GroupNorm.onnx inputshape=[1,12,64],[1,12,24,64],[1,12,24,32,64]")

    # pnnx inference
    import test_nn_GroupNorm_pnnx
    b0, b1, b2 = test_nn_GroupNorm_pnnx.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4) and torch.allclose(a2, b2, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
