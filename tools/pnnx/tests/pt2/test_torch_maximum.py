# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.maximum(x, y)
        out1 = torch.maximum(y, y)
        out2 = torch.maximum(z, torch.ones_like(z) + 0.1)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a = net(x, y, z)

    # export pt2
    mod = torch.export.export(net, (x, y, z))
    torch.export.save(mod, "test_torch_maximum.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_torch_maximum.pt2 inputshape=[3,16],[3,16],[5,9,3]")

    # pnnx inference
    import test_torch_maximum_pnnx
    b = test_torch_maximum_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
