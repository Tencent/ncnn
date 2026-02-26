# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = F.adaptive_max_pool3d(x, output_size=(7,6,5))
        out1 = F.adaptive_max_pool3d(x, output_size=1)
        out2 = F.adaptive_max_pool3d(x, output_size=(None,4,3))
        out3 = F.adaptive_max_pool3d(x, output_size=(5,None,None))
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 33, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_adaptive_max_pool3d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_max_pool3d.pt inputshape=[1,12,24,33,64]")

    # ncnn inference
    import test_F_adaptive_max_pool3d_ncnn
    b = test_F_adaptive_max_pool3d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        b0 = b0.reshape_as(a0)
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
