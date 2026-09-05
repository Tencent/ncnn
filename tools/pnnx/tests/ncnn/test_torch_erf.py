# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.erf(x * 2 - 1)
        y = torch.erf(y * 2 - 1)
        z = torch.erf(z * 2 - 1)
        w = F.max_pool2d(w, 1)
        w = torch.erf(w * 2 - 1)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(3, 12, 16)
    z = torch.rand(2, 3, 4, 5)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_erf.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_erf.pt inputshape=[16],[3,12,16],[2,3,4,5],[2,3,5,7]")

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
