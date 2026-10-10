# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        a = torch.ge(x, y)
        b = torch.ne(x, y)
        out = torch.bitwise_or(a, b)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_bitwise_or.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_bitwise_or.pt inputshape=[3,16],[3,16]")

    # pnnx inference
    import test_torch_bitwise_or_pnnx
    b = test_torch_bitwise_or_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
