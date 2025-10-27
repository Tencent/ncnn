# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.unflatten(dim=2, sizes=(2,1,2,-1))
        y = y.unflatten(dim=1, sizes=(1,1,5))
        z = z.unflatten(dim=-2, sizes=(3,-1))
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('1.13'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_unflatten.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_unflatten.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_Tensor_unflatten_pnnx
    b = test_Tensor_unflatten_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
