# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q, r):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        q = F.max_pool2d(q, 1)
        r = F.max_pool1d(r, 1)
        q0 = q.reshape_as(r)
        q1 = r.reshape_as(q)
        x = x + 1
        y = y - 1
        z = z * 2
        q0 = q0 + 1
        q1 = q1 - 1
        return x, y, z, q0, q1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)
    q = torch.rand(2, 3, 5, 7)
    r = torch.rand(2, 3, 35)

    a = net(x, y, z, q, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q, r))
    mod.save("test_Tensor_reshape_as.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_Tensor_reshape_as.pt inputshape=[1,3,16],[6,2,2,2],[48],[2,3,5,7],[2,3,35]")

    # ncnn inference
    import test_Tensor_reshape_as_ncnn
    b = test_Tensor_reshape_as_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
