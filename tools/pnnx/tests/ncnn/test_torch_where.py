# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.where(x > 0.5, y, z)
        out1 = torch.where(x > 0.3, y, y)
        out2 = torch.where(z > 0.5, y, z)
        out3 = torch.where(x > 0.5, 1.0, z)
        out4 = torch.where(x > 0.5, y, 0.0)
        out5 = torch.where(x > 0.5, 1.0, 0.0)
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(3, 16)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_where.pt")

    # torchscript to pnnx
    import os
    os.system("../../build/src/pnnx.exe test_torch_where.pt inputshape=[3,16],[3,16],[3,16]")

    # ncnn inference
    import test_torch_where_ncnn
    b = test_torch_where_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
