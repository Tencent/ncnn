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

    def forward(self, x, y):
        x = self.ln_0(x)
        y = self.ln_0(y)
        z = self.ln_1(y)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_LayerNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_LayerNorm.pt inputshape=[1,24,64],[1,12,24,64]")

    # ncnn inference
    import test_nn_LayerNorm_ncnn
    b = test_nn_LayerNorm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
