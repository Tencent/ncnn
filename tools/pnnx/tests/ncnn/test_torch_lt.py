# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.lt(x, y)
        out1 = torch.lt(y, y)
        out2 = torch.lt(z, 1)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_lt.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_lt.pt inputshape=[3,16],[3,16],[5,9,3]")

    # ncnn inference
    import test_torch_lt_ncnn
    b = test_torch_lt_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0.bool()):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
