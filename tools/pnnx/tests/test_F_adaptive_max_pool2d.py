# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0, indices0 = F.adaptive_max_pool2d(x, output_size=(7,6), return_indices=True)
        out1 = F.adaptive_max_pool2d(x, output_size=1)
        out2 = F.adaptive_max_pool2d(x, output_size=(None,3))
        out3, indices3 = F.adaptive_max_pool2d(x, output_size=(5,None), return_indices=True)
        return out0, indices0, out1, out2, out3, indices3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_adaptive_max_pool2d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_adaptive_max_pool2d.pt inputshape=[1,12,24,64]")

    # pnnx inference
    import test_F_adaptive_max_pool2d_pnnx
    b = test_F_adaptive_max_pool2d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
