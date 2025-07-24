# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = F.fold(x, output_size=22, kernel_size=3)
        y = F.fold(y, output_size=(17,18), kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        z = F.fold(z, output_size=(5,11), kernel_size=(2,3), stride=1, padding=(2,4), dilation=(1,2))

        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 108, 400)
    y = torch.rand(1, 96, 190)
    z = torch.rand(1, 36, 120)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_fold.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_fold.pt inputshape=[1,108,400],[1,96,190],[1,36,120]")

    # pnnx inference
    import test_F_fold_pnnx
    b0, b1, b2 = test_F_fold_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
