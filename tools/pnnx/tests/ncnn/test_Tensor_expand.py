# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.expand(24)
        y = y.expand(-1, 11, -1)
        z = z.expand(8, 3, -1, 4)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1)
    y = torch.rand(3, 1, 1)
    z = torch.rand(8, 1, 9, 1)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_expand.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_expand.pt inputshape=[1],[3,1,1],[8,1,9,1]")

    # ncnn inference
    import test_Tensor_expand_ncnn
    b = test_Tensor_expand_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
