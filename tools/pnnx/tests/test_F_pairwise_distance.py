# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        z1 = F.pairwise_distance(x,y,p=1,keepdim=False)
        z2 = F.pairwise_distance(x,y,p=2,keepdim=True)
        z3 = F.pairwise_distance(x,y)
        z4 = F.pairwise_distance(x,y,eps = 1e-3)
        return z1,z2,z3,z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 128, 128)
    y = torch.rand(12, 128, 128)

    a0,a1,a2,a3 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_pairwise_distance.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_pairwise_distance.pt inputshape=[12,128,128],[12,128,128]")

    # pnnx inference
    import test_F_pairwise_distance_pnnx
    b0,b1,b2,b3 = test_F_pairwise_distance_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_F_pairwise_distance_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_pairwise_distance_export.pt2 fp16=0")

    # pnnx inference
    import test_F_pairwise_distance_export_pnnx
    c0, c1, c2, c3 = test_F_pairwise_distance_export_pnnx.test_inference()

    return torch.equal(a0,c0) and torch.equal(a1,c1) and torch.equal(a2,c2) and torch.equal(a3,c3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
