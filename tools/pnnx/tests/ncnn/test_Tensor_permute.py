# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.permute(1, 0)
        x = x.permute(0, 1)
        y = y.permute(2, 1, 0)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 3, 0, 2)
        z = z.permute(2, 0, 3, 1)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_permute.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_permute.pt inputshape=[3,16],[5,9,11],[8,5,9,10]")

    # ncnn inference
    import test_Tensor_permute_ncnn
    b = test_Tensor_permute_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
