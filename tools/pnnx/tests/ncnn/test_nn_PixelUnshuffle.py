# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.down_0 = nn.PixelUnshuffle(2)
        self.down_1 = nn.PixelUnshuffle(4)

    def forward(self, x):
        x = self.down_0(x)
        x = self.down_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)

    a0 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_PixelUnshuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_PixelUnshuffle.pt inputshape=[1,3,128,128]")

    # ncnn inference
    import test_nn_PixelUnshuffle_ncnn
    b0 = test_nn_PixelUnshuffle_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
