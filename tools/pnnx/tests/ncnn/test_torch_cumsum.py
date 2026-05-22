# Copyright 2021 Tencent
# Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        # x - 3d
        # y - 2d
        # z - 1d
        # w - 4d
        x0 = torch.cumsum(x, dim=0)
        x1 = torch.cumsum(x, dim=1)
        x2 = torch.cumsum(x, dim=2)

        y0 = torch.cumsum(y, dim=0)
        y1 = torch.cumsum(y, dim=1)

        z0 = torch.cumsum(z, dim=0)

        w0 = torch.cumsum(w, dim=0)
        w1 = torch.cumsum(w, dim=1)
        w2 = torch.cumsum(w, dim=2)
        w3 = torch.cumsum(w, dim=3)
        w4 = torch.cumsum(w, dim=-1)
        return x0, x1, x2, y0, y1, z0, w0, w1, w2, w3, w4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 16)
    y = torch.rand(5, 9)
    z = torch.rand(3)
    w = torch.rand(2, 3, 4, 5)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_cumsum.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_cumsum.pt inputshape=[2,3,16],[5,9],[3],[2,3,4,5]")

    # ncnn inference
    import test_torch_cumsum_ncnn
    b = test_torch_cumsum_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
