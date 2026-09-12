# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.linear_1 = nn.Linear(in_features=16, out_features=3, bias=True)

    def forward(self, x, y, z):
        x = self.linear_0(x)
        x = self.linear_1(x)

        y = self.linear_0(y)
        y = self.linear_1(y)

        z = self.linear_0(z)
        z = self.linear_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64)
    y = torch.rand(12, 64)
    z = torch.rand(1, 3, 12, 64)

    a0, a1, a2 = net(x, y, z)

    program = torch.export.export(net, (x, y, z))
    torch.export.save(program, "test_nn_Linear.pt2")

    from pnnxutil import run_pnnx
    run_pnnx("test_nn_Linear.pt2", "inputshape=[1,64],[12,64],[1,3,12,64]")

    import test_nn_Linear_pnnx
    b0, b1, b2 = test_nn_Linear_pnnx.test_inference()

    return torch.allclose(a0, b0, 1e-3, 1e-3) and torch.allclose(a1, b1, 1e-3, 1e-3) and torch.allclose(a2, b2, 1e-3, 1e-3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
