# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q):
        x = F.pad(x, (3,4), mode='constant', value=1.3)
        x = F.pad(x, (2,2))

        y = F.pad(y, (5,6), mode='reflect')
        y = F.pad(y, (2,1), mode='replicate')
        y = F.pad(y, (3,4), mode='constant', value=1.3)
        y = F.pad(y, (1,1))

        z = F.pad(z, (3,4,3,4), mode='reflect')
        z = F.pad(z, (2,1,2,0), mode='replicate')
        z = F.pad(z, (1,0,2,0), mode='constant', value=1.3)
        z = F.pad(z, (3,3,3,3))
        q = F.max_pool2d(q, 1)
        q = F.pad(q, (1,2,2,1), mode='constant', value=1.3)

        if version.parse(torch.__version__) < version.parse('1.10'):
            w = w.relu()
            return x, y, z, w, q

        w = F.pad(w, (1,2,2,1,1,0), mode='reflect')
        w = F.pad(w, (2,1,1,0,0,1), mode='replicate')
        w = F.pad(w, (1,0,2,0,0,1), mode='constant', value=1.3)
        w = F.pad(w, (1,1,2,2,1,0))

        return x, y, z, w, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(1, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(3, 4, 12, 16)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q))
    mod.save("test_F_pad.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_pad.pt inputshape=[1,16],[1,2,16],[1,3,12,16],[3,4,12,16],[2,3,5,7]")

    # ncnn inference
    import test_F_pad_ncnn
    b = test_F_pad_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
