# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    os.system(os.path.join("..", "src", "pnnx") + " test_F_pairwise_distance.pt inputshape=[12,128,128],[12,128,128]")

    # pnnx inference
    import test_F_pairwise_distance_pnnx
    b0,b1,b2,b3 = test_F_pairwise_distance_pnnx.test_inference()

    ts_ok = torch.equal(a0,b0) and torch.equal(a1,b1) and torch.equal(a2,b2) and torch.equal(a3,b3)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y), ["[12,128,128]", "[12,128,128]"], "test_F_pairwise_distance")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
