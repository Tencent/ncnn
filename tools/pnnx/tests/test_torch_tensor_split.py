# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x0, x1, x2 = torch.tensor_split(x, (12, 13))
        y0, y1 = torch.tensor_split(y, 2, dim=0)
        y2, y3, y4 = torch.tensor_split(y, 3, dim=1)
        z0, z1 = torch.tensor_split(z, (3,), dim=0)
        z2, z3 = torch.tensor_split(z, (1,), dim=1)
        z4, z5, z6 = torch.tensor_split(z, 3, dim=2)
        w0, w1, w2 = torch.tensor_split(w, (2, 4), dim=0)
        w3, w4 = torch.tensor_split(w, 2, dim=1)
        w5, w6, w7 = torch.tensor_split(w, (1, 5), dim=2)
        w8, w9, wa, wb, wc = torch.tensor_split(w, (1, 3, 7, 17), dim=3)
        return x0, x1, x2, y0, y1, y2, y3, y4, z0, z1, z2, z3, z4, z5, z6, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(100)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)
    w = torch.rand(6, 13, 6, 22)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_tensor_split.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_tensor_split.pt inputshape=[100],[3,16],[5,9,3],[6,13,6,22]")

    # pnnx inference
    import test_torch_tensor_split_pnnx
    b = test_torch_tensor_split_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
