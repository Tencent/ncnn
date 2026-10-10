# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.c0 = nn.Parameter(torch.rand(12))
        self.c2 = nn.Parameter(torch.rand(48, 12))

    def forward(self, a0, a1, a2, b0, b1, b2, c1):
        a = torch.addmm(a0, a1, a2)
        b = torch.addmm(b0, b1, b2, beta=1.4, alpha=0.7)
        c = torch.addmm(self.c0, c1, self.c2, beta=1, alpha=1)
        return a, b, c

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(23)
    a1 = torch.rand(13, 16)
    a2 = torch.rand(16, 23)
    b0 = torch.rand(7, 33)
    b1 = torch.rand(7, 26)
    b2 = torch.rand(26, 33)
    c1 = torch.rand(16, 48)

    a = net(a0, a1, a2, b0, b1, b2, c1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1, a2, b0, b1, b2, c1))
    mod.save("test_torch_addmm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_addmm.pt inputshape=[23],[13,16],[16,23],[7,33],[7,26],[26,33],[16,48]")

    # ncnn inference
    import test_torch_addmm_ncnn
    b = test_torch_addmm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
