# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = F.pixel_shuffle(x, 4)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 6, 7)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_pixel_shuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_pixel_shuffle.pt inputshape=[1,128,6,7]")

    # pnnx inference
    import test_F_pixel_shuffle_pnnx
    b = test_F_pixel_shuffle_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
