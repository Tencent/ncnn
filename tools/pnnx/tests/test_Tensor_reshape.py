# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(3, 4, 1)
        self.conv3 = nn.Conv2d(4, 3, 1)
        self.conv4 = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w, u, v, r, s):
        x = x.reshape(1, 2, 24)
        x = x.reshape(48)
        y = y.reshape(1, 11, 5, 9)
        y = y.reshape(99, 5)
        z = z.reshape(4, 3, 30, 10, 14)
        z = z.reshape(15, 2, 10, 7, 8, 3)
        w = w.reshape(2, 3, 5, 7)
        w = self.conv(w)
        u = self.conv2(u)
        u = u.permute(1, 0, 2, 3)
        u = u.reshape(8, 5, 7)
        v = v.reshape(4, 2, 5, 7)
        v = v.permute(1, 0, 2, 3)
        v = self.conv3(v)
        r = self.conv4(r)
        r = r.reshape(2, -1)
        s = s.reshape(2, 3, -1, 7)
        s = self.conv(s)
        return x, y, z, w, u, v, r, s

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    w = torch.rand(210)
    u = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)
    r = torch.rand(2, 3, 5, 7)
    s = torch.rand(210)

    a = net(x, y, z, w, u, v, r, s)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, u, v, r, s))
    mod.save("test_Tensor_reshape.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_reshape.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10],[210],[2,3,5,7],[280],[2,3,5,7],[210]")

    # pnnx inference
    import test_Tensor_reshape_pnnx
    b = test_Tensor_reshape_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
