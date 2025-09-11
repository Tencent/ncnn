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
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w = F.relu(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(12, 64)
    z = torch.rand(12, 24, 64)
    w = torch.rand(12, 24, 32, 64)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_nn_AlphaDropout.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_AlphaDropout.pt inputshape=[12],[12,64],[12,24,64],[12,24,32,64]")

    # ncnn inference
    import test_nn_AlphaDropout_ncnn
    b = test_nn_AlphaDropout_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
