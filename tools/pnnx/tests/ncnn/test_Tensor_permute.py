# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 3, 1)

    def forward(self, x, y, z, w, v):
        x = x.permute(1, 0)
        x = x.permute(0, 1)
        y = y.permute(2, 1, 0)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 3, 0, 2)
        z = z.permute(2, 0, 3, 1)
        w = self.conv(w)
        w0 = w.permute(1, 0, 2, 3).reshape(8, 5, 7)
        w1 = w.permute(1, 0, 3, 2).reshape(8, 7, 5)
        v = v.reshape(4, 2, 5, 7)
        v = v.permute(1, 0, 2, 3)
        v = self.conv2(v)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z, w0, w1, v

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)

    a = net(x, y, z, w, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, v))
    mod.save("test_Tensor_permute.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_permute.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7],[280]")

    with open("test_Tensor_permute.ncnn.param") as f:
        lines = f.readlines()
        if sum(1 for line in lines if line.startswith("Reshape") and "12=1" in line) != 2:
            return False
        if sum(1 for line in lines if line.startswith("Reshape") and "12=2" in line) != 1:
            return False
        if sum(1 for line in lines if line.startswith("Permute")) != 5:
            return False

    # ncnn inference
    import test_Tensor_permute_ncnn
    b = test_Tensor_permute_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
