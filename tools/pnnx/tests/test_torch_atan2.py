# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.atan2(x, y)
        out1 = torch.atan2(y, y)
        out2 = torch.atan2(z, torch.ones_like(z) + 0.5)
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
    mod.save("test_torch_atan2.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_atan2.pt inputshape=[3,16],[3,16],[5,9,3]")

    # pnnx inference
    import test_torch_atan2_pnnx
    b = test_torch_atan2_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
