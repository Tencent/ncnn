# Copyright 2024 Tencent
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

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_nn_Linear.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_Linear.onnx inputshape=[1,64],[12,64],[1,3,12,64]")

    # pnnx inference
    import test_nn_Linear_pnnx
    b0, b1, b2 = test_nn_Linear_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
