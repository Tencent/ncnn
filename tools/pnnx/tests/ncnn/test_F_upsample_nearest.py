# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, w):
        x = F.upsample_nearest(x, size=(12,12))
        x = F.upsample_nearest(x, scale_factor=2)

        w = F.upsample_nearest(w, scale_factor=(2.976744,2.976744))
        return x, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, w))
    mod.save("test_F_upsample_nearest.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_upsample_nearest.pt inputshape=[1,12,24,64],[1,8,86,86]")

    # ncnn inference
    import test_F_upsample_nearest_ncnn
    b = test_F_upsample_nearest_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
