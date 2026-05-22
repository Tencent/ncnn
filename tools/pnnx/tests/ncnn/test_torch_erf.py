# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.erf(x * 2 - 1)
        y = torch.erf(y * 2 - 1)
        z = torch.erf(z * 2 - 1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(3, 12, 16)
    z = torch.rand(2, 3, 4, 5)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_erf.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_erf.pt inputshape=[16],[3,12,16],[2,3,4,5]")

    # ncnn inference
    import test_torch_erf_ncnn
    b = test_torch_erf_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
