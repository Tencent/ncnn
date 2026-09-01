# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x[:2,:].fill_(z[0])
        y[:1,:].fill_(0.22)
        return x + y.fill_(7)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)
    z = torch.rand(1)

    a = net(x, y, z)

    # export pt2
    mod = torch.export.export(net, (x, y, z))
    torch.export.save(mod, "test_Tensor_fill.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_Tensor_fill.pt2 inputshape=[6,16],[6,16],[1]")

    # pnnx inference
    import test_Tensor_fill_pnnx
    b = test_Tensor_fill_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
