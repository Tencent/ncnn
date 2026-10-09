# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.round(x * 10)
        y = torch.round(y * 10)
        z = torch.round(z * 10)
        w = F.max_pool2d(w, 1)
        w = torch.round(w * 10)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_round.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_round.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_round_ncnn
    b = test_torch_round_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
