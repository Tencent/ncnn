# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        a = x > 0.5
        b = y > 0.5
        out0 = torch.logical_not(a)
        out1 = torch.logical_and(a, b)
        out2 = torch.logical_or(a, b)
        out3 = torch.logical_xor(a, b)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_logical.pt")

    # torchscript to pnnx
    import os
    os.system("../../build/src/pnnx.exe test_torch_logical.pt inputshape=[3,16],[3,16]")

    # ncnn inference
    import test_torch_logical_ncnn
    b = test_torch_logical_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
