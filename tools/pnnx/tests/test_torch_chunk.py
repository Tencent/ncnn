# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        y0, y1, y2 = torch.chunk(y, chunks=3, dim=2)
        z0, z1, z2, z3, z4 = torch.chunk(z, chunks=5, dim=0)
        return x0, x1, y0, y1, y2, z0, z1, z2, z3, z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_chunk.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_chunk.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_torch_chunk_pnnx
    b = test_torch_chunk_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
