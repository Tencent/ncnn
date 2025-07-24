# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x * 100
        z = z * 200
        x = x.type_as(y)
        x = F.relu(x)
        x = x.type_as(z)
        z = F.relu(z)
        z = z.type_as(x)
        return x, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.randint(10, (1, 13), dtype=torch.int)
    z = torch.rand(8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_type_as.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_type_as.pt inputshape=[3,16],[1,13]i32,[8,5,9,10]")

    # pnnx inference
    import test_Tensor_type_as_pnnx
    b = test_Tensor_type_as_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
