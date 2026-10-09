# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x0, x1 = torch.split(x, split_size_or_sections=2, dim=1)
        y0, y1, y2 = torch.split(y, split_size_or_sections=[1,3,5], dim=2)
        z0, z1, z2, z3, z4 = torch.split(z, split_size_or_sections=3, dim=0)
        q = F.max_pool2d(q, 1)
        q0, q1, q2 = torch.split(q, split_size_or_sections=[1,2,3], dim=1)
        q3, q4 = torch.split(q, split_size_or_sections=4, dim=-1)
        return x0, x1, y0, y1, y2, z0, z1, z2, z3, z4, q0, q1, q2, q3, q4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9)
    q = torch.rand(2, 6, 5, 8)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_torch_split.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_split.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9],[2,6,5,8]")

    # ncnn inference
    import test_torch_split_ncnn
    b = test_torch_split_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
