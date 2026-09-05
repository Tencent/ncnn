# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x = F.normalize(x, dim=0)
        x = F.normalize(x, dim=0, eps=1e-3)

        y = F.normalize(y, dim=0)
        y = F.normalize(y, dim=0, eps=1e-4)

        z = F.normalize(z, dim=0)
        z = F.normalize(z, dim=0, eps=1e-4)
        q = F.max_pool2d(q, 1)
        q0 = F.normalize(q, dim=1)
        q1 = F.normalize(q, dim=1, eps=1e-4)
        return x, y, z, q0, q1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(64)
    y = torch.rand(12, 24, 64)
    z = torch.rand(12, 3, 24, 64)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_F_normalize.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_normalize.pt inputshape=[64],[12,24,64],[12,3,24,64],[2,3,5,7]")

    # ncnn inference
    import test_F_normalize_ncnn
    b = test_F_normalize_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
