# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q):
        x = x.expand(24)
        y = y.expand(-1, 11, -1)
        z = z.expand(8, 3, -1, 4)
        w = F.max_pool2d(w, 1)
        w = w.expand(-1, -1, 5, 7)
        q = q.expand(2, 1, 3, 5)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w = F.relu(w)
        q = F.relu(q)
        return x, y, z, w, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1)
    y = torch.rand(3, 1, 1)
    z = torch.rand(8, 1, 9, 1)
    w = torch.rand(2, 3, 1, 1)
    q = torch.rand(1, 3, 1)

    a = net(x, y, z, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q))
    mod.save("test_Tensor_expand.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_expand.pt inputshape=[1],[3,1,1],[8,1,9,1],[2,3,1,1],[1,3,1]")

    # ncnn inference
    import test_Tensor_expand_ncnn
    b = test_Tensor_expand_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
