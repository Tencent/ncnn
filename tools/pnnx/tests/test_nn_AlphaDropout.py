# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.AlphaDropout()
        self.dropout_1 = nn.AlphaDropout(p=0.7)

    def forward(self, x, y, z, w):
        x = self.dropout_0(x)
        y = self.dropout_0(y)
        z = self.dropout_1(z)
        w = self.dropout_1(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 12, 24, 32, 64)

    a0, a1, a2, a3 = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_nn_AlphaDropout.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AlphaDropout.pt inputshape=[1,12],[1,12,64],[1,12,24,64],[1,12,24,32,64]")

    # pnnx inference
    import test_nn_AlphaDropout_pnnx
    b0, b1, b2, b3 = test_nn_AlphaDropout_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
