# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a0, a1):
        a = torch.mm(a0, a1)
        return a

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(23, 14)
    a1 = torch.rand(14, 35)

    a = net(a0, a1)

    # export pt2
    mod = torch.export.export(net, (a0, a1))
    torch.export.save(mod, "test_torch_mm.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_torch_mm.pt2 inputshape=[23,14],[14,35]")

    # pnnx inference
    import test_torch_mm_pnnx
    b = test_torch_mm_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
