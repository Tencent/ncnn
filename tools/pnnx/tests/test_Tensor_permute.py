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
        x = x.permute(1, 0, 2)
        x = x.permute(0, 1, 2)
        y = y.permute(2, 3, 1, 0)
        y = y.permute(3, 1, 0, 2)
        z = z.permute(1, 3, 0, 4, 2)
        z = z.permute(0, 2, 4, 3, 1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_permute.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_permute.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_Tensor_permute_pnnx
    b = test_Tensor_permute_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
