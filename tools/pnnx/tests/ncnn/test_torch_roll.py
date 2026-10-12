# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x = torch.roll(x, 3, 1)
        y = torch.roll(y, -2, -1)
        z = torch.roll(z, shifts=(2,1), dims=(0,1))
        q = F.max_pool2d(q, 1)
        q0 = torch.roll(q, 2, 1)
        q1 = torch.roll(q, shifts=(1,-2), dims=(2,3))
        return x, y, z, q0, q1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_torch_roll.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_torch_roll.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_roll_ncnn
    b = test_torch_roll_ncnn.test_inference()

    print(x)
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            print(a0)
            print(b0)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
