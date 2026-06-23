# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LocalResponseNorm(3)
        self.ln_1 = nn.LocalResponseNorm(size=5, alpha=0.001, beta=0.8, k=0.9)

    def forward(self, x, y):
        x = self.ln_0(x)
        x = self.ln_1(x)
        y = self.ln_0(y)
        y = self.ln_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    y = torch.rand(2, 12, 24, 64)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_LocalResponseNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_LocalResponseNorm.pt inputshape=[1,12,24,64],[2,12,24,64]")

    # ncnn inference
    import test_nn_LocalResponseNorm_ncnn
    b = test_nn_LocalResponseNorm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
