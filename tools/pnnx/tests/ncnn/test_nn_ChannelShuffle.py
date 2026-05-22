# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.shuffle_0 = nn.ChannelShuffle(2)
        self.shuffle_1 = nn.ChannelShuffle(16)

    def forward(self, x, y, z):
        x = self.shuffle_0(x)
        x = self.shuffle_1(x)

        y = self.shuffle_0(y)
        y = self.shuffle_1(y)

        z = self.shuffle_0(z)
        z = self.shuffle_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64, 6, 8)
    y = torch.rand(1, 96, 7, 9)
    z = torch.rand(1, 64, 3, 6, 8)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_ChannelShuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_ChannelShuffle.pt inputshape=[1,64,6,8],[1,96,7,9],[1,64,3,6,8]")

    # ncnn inference
    import test_nn_ChannelShuffle_ncnn
    b0, b1, b2 = test_nn_ChannelShuffle_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4) and torch.allclose(a2, b2, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
