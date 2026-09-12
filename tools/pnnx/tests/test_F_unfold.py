# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x0 = F.unfold(x, kernel_size=3)
        x1 = F.unfold(x, kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        x2 = F.unfold(x, kernel_size=(1,3), stride=1, padding=(2,4), dilation=(1,2))

        return x0, x1, x2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)

    a0, a1, a2 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_unfold.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_F_unfold.pt inputshape=[1,12,64,64]")

    # pnnx inference
    import test_F_unfold_pnnx
    b0, b1, b2 = test_F_unfold_pnnx.test_inference()

    ts_ok = torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,12,64,64]"], "test_F_unfold")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
