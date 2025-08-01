# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.linear_1 = nn.Linear(in_features=16, out_features=13, bias=True)

        self.linear_2 = nn.Linear(in_features=13, out_features=17, bias=True)
        self.linear_2 = torch.nn.utils.weight_norm(self.linear_2)

    def forward(self, x, y, z):
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.linear_2(x)

        y = self.linear_0(y)
        y = self.linear_1(y)
        y = self.linear_2(y)

        z = self.linear_0(z)
        z = self.linear_1(z)
        z = self.linear_2(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64)
    y = torch.rand(12, 64)
    z = torch.rand(1, 3, 12, 64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_Linear.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_Linear.pt inputshape=[1,64],[12,64],[1,3,12,64]")

    # pnnx inference
    import test_nn_Linear_pnnx
    b = test_nn_Linear_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        b0 = b0.reshape_as(a0)
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
