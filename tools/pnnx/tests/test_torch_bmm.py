# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a0, a1):
        a = torch.bmm(a0, a1)
        return a

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(10, 23, 14)
    a1 = torch.rand(10, 14, 5)

    a = net(a0, a1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1))
    mod.save("test_torch_bmm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_bmm.pt inputshape=[10,23,14],[10,14,5]")

    # pnnx inference
    import test_torch_bmm_pnnx
    b = test_torch_bmm_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
