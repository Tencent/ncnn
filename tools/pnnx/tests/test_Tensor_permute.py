# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 3, 1)
        self.conv3 = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w, v):
        x = x.permute(1, 0, 2)
        x = x.permute(0, 1, 2)
        y = y.permute(2, 3, 1, 0)
        y = y.permute(3, 1, 0, 2)
        z = z.permute(1, 3, 0, 4, 2)
        z = z.permute(0, 2, 4, 3, 1)
        wb = self.conv(w)
        w0 = wb.permute(1, 0, 2, 3)
        w1 = wb.permute(2, 0, 1, 3)
        w = self.conv3(w)
        w2 = w.permute(1, 0, 2, 3).reshape(8, 5, 7)
        w3 = w.permute(1, 0, 3, 2).reshape(8, 7, 5)
        v = v.reshape(4, 2, 5, 7)
        v = v.permute(1, 0, 2, 3)
        v = self.conv2(v)
        return x, y, z, w0, w1, w2, w3, v

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)

    a = net(x, y, z, w, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, v))
    mod.save("test_Tensor_permute.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_permute.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10],[2,3,5,7],[280]")

    # pnnx inference
    import test_Tensor_permute_pnnx
    b = test_Tensor_permute_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
