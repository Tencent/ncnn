# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w):
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(y, 2, 3)
        z = torch.transpose(z, 1, 3)
        w = self.conv(w)
        w0 = torch.transpose(w, 0, 1)
        w1 = torch.transpose(w, 0, 2)
        return x, y, z, w0, w1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_transpose.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_transpose.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10],[2,3,5,7]")

    with open("test_torch_transpose.ncnn.param") as f:
        if sum(1 for line in f.readlines() if line.startswith("Permute")) != 3:
            return False

    # pnnx inference
    import test_torch_transpose_pnnx
    b = test_torch_transpose_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
