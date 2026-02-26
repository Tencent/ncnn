# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.reshape(1, 2, 24)
        x = x.reshape(48)
        y = y.reshape(1, 11, 5, 9)
        y = y.reshape(99, 5)
        z = z.reshape(4, 3, 30, 10, 14)
        z = z.reshape(15, 2, 10, 7, 8, 3)
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
    mod.save("test_Tensor_reshape.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_reshape.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_Tensor_reshape_pnnx
    b = test_Tensor_reshape_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
