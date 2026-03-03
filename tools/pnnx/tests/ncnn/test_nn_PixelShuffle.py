# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.up_0 = nn.PixelShuffle(4)
        self.up_1 = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.up_0(x)
        x = self.up_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 6, 8)

    a0 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_PixelShuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_PixelShuffle.pt inputshape=[1,128,6,8]")

    # ncnn inference
    import test_nn_PixelShuffle_ncnn
    b0 = test_nn_PixelShuffle_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
