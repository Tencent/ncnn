# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q):
        x = F.softplus(x)
        y = F.softplus(y, threshold=12)
        z = F.softplus(z)
        w = F.softplus(w)
        q = F.max_pool2d(q, 1)
        q = F.softplus(q)
        return x, y, z, w, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(3, 12, 16)
    z = torch.rand(2, 3, 4, 5)
    w = torch.rand(2, 3, 5, 7)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q))
    mod.save("test_F_softplus.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_softplus.pt inputshape=[16],[3,12,16],[2,3,4,5],[2,3,5,7],[2,3,5,7]")

    # ncnn inference
    import test_F_softplus_ncnn
    b = test_F_softplus_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
