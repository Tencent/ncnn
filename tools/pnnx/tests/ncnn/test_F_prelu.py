# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w4 = nn.Parameter(torch.rand(16))
        self.w5 = nn.Parameter(torch.rand(2))
        self.w6 = nn.Parameter(torch.rand(3))
        self.w7 = nn.Parameter(torch.rand(12))

    def forward(self, x, y, z):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        x = F.prelu(x, self.w4)
        y = F.prelu(y, self.w5)
        z = F.prelu(z, self.w6)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(1, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    # w = torch.rand(1, 5, 7, 9, 11)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_prelu.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_prelu.pt inputshape=[1,16],[1,2,16],[1,3,12,16]")

    # ncnn inference
    import test_F_prelu_ncnn
    b = test_F_prelu_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
