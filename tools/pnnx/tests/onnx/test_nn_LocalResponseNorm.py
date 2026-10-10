# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LocalResponseNorm(3)
        self.ln_1 = nn.LocalResponseNorm(size=5, alpha=0.001, beta=0.8, k=0.9)

    def forward(self, x, y, z):
        x = self.ln_0(x)
        x = self.ln_1(x)

        y = self.ln_0(y)
        y = self.ln_1(y)

        z = self.ln_0(z)
        z = self.ln_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2 = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_nn_LocalResponseNorm.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_LocalResponseNorm.onnx inputshape=[1,24,64],[1,12,24,64],[1,12,16,24,64]")

    # pnnx inference
    import test_nn_LocalResponseNorm_pnnx
    b0, b1, b2 = test_nn_LocalResponseNorm_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
