# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x.select(1, 1)
        y = y.select(2, 4)
        z0 = z.select(-1, 0)
        z1 = z.select(-1, 1)
        w = F.max_pool2d(w, 1)
        w0 = w.select(1, 1)
        w1 = w.select(-1, 3)
        return x, y, z0, z1, w0, w1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(4, 5, 2)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_Tensor_select.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_select.pt inputshape=[1,3,16],[1,5,9,11],[4,5,2],[2,3,5,7]")

    # ncnn inference
    import test_Tensor_select_ncnn
    b = test_Tensor_select_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
