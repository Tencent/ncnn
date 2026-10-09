# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x0, x1 = torch.chunk(x, chunks=2, dim=0)
        y0, y1, y2 = torch.chunk(y, chunks=3, dim=1)
        z0, z1, z2, z3, z4 = torch.chunk(z, chunks=5, dim=1)
        w = F.max_pool2d(w, 1)
        w0, w1 = torch.chunk(w, chunks=2, dim=1)
        w2, w3 = torch.chunk(w, chunks=2, dim=-1)
        return x0, x1, y0, y1, y2, z0, z1, z2, z3, z4, w0, w1, w2, w3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(4, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(14, 5, 9, 10)
    w = torch.rand(2, 6, 5, 8)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_chunk.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_chunk.pt inputshape=[4,16],[5,9,11],[14,5,9,10],[2,6,5,8]")

    # ncnn inference
    import test_torch_chunk_ncnn
    b = test_torch_chunk_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
