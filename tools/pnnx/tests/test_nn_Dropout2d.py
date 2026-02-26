# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.Dropout2d()
        self.dropout_1 = nn.Dropout2d(p=0.7)

    def forward(self, x, y):
        x = self.dropout_0(x)
        y = self.dropout_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    y = torch.rand(1, 3, 4, 5)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_Dropout2d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_Dropout2d.pt inputshape=[1,12,24,64],[1,3,4,5]")

    # pnnx inference
    import test_nn_Dropout2d_pnnx
    b0, b1 = test_nn_Dropout2d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
