# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LayerNorm(64)
        self.ln_0.weight = nn.Parameter(torch.rand(64))
        self.ln_0.bias = nn.Parameter(torch.rand(64))
        self.ln_1 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)

        self.ln_3 = nn.LayerNorm(64)
        self.ln_3.weight = nn.Parameter(torch.rand(64))
        self.ln_3.bias = None
        self.ln_4 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-3)
        self.ln_4.weight = nn.Parameter(torch.rand(24,64))
        self.ln_4.bias = None

    def forward(self, x, y, z):
        x = self.ln_0(x)
        x = self.ln_1(x)

        y = self.ln_0(y)
        y = self.ln_1(y)

        z = self.ln_0(z)
        z = self.ln_1(z)
        u = self.ln_3(x)
        v = self.ln_4(y)
        return x, y, z, u, v

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2, a3, a4 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_LayerNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_LayerNorm.pt inputshape=[1,24,64],[1,12,24,64],[1,12,16,24,64]")

    # pnnx inference
    import test_nn_LayerNorm_pnnx
    b0, b1, b2, b3, b4 = test_nn_LayerNorm_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3) and torch.equal(a4, b4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
