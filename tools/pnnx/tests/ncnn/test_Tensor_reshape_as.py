# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q, r, s, t, u, v):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        q = F.max_pool2d(q, 3, stride=1, padding=1)
        r = F.max_pool1d(r, 1)
        q0 = q.reshape_as(r)
        q1 = r.reshape_as(q)
        s = F.max_pool2d(s, 1)
        t = F.max_pool2d(t, 1)
        t = t.permute(1, 0, 2, 3)
        s = s.reshape_as(t)
        u = F.max_pool2d(u, 1)
        v = F.max_pool2d(v, 1)
        v = v.permute(1, 0, 2, 3)
        u = u.reshape_as(v)
        x = x + 1
        y = y - 1
        z = z * 2
        q0 = q0 + 1
        q1 = q1 - 1
        s = s + 2
        u = u + 3
        return x, y, z, q0, q1, s, u

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)
    q = torch.rand(2, 3, 5, 7)
    r = torch.rand(2, 3, 35)
    s = torch.rand(2, 3, 5, 7)
    t = torch.rand(2, 3, 5, 7)
    u = torch.rand(2, 3, 5, 7)
    v = torch.rand(3, 2, 5, 7)

    a = net(x, y, z, q, r, s, t, u, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q, r, s, t, u, v))
    mod.save("test_Tensor_reshape_as.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_Tensor_reshape_as.pt inputshape=[1,3,16],[6,2,2,2],[48],[2,3,5,7],[2,3,35],[2,3,5,7],[2,3,5,7],[2,3,5,7],[3,2,5,7] inputshape2=[1,3,16],[6,2,2,2],[48],[2,3,5,7],[2,3,35],[2,3,5,7],[2,3,5,7],[4,6,8,10],[6,4,8,10]")

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
