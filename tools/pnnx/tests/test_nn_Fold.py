# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fold_0 = nn.Fold(output_size=22, kernel_size=3)
        self.fold_1 = nn.Fold(output_size=(17,18), kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        self.fold_2 = nn.Fold(output_size=(5,11), kernel_size=(2,3), stride=1, padding=(2,4), dilation=(1,2))

    def forward(self, x, y, z):
        x = self.fold_0(x)
        y = self.fold_1(y)
        z = self.fold_2(z)

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
    mod.save("test_nn_Fold.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_nn_Fold.pt inputshape=[1,108,400],[1,96,190],[1,36,120]")

    # pnnx inference
    import test_nn_Fold_pnnx
    b0, b1, b2 = test_nn_Fold_pnnx.test_inference()

    # fold sums overlapping windows; the accumulation order can differ across
    # platforms/threads, so compare with tolerance instead of exact equality
    ts_ok = torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4) and torch.allclose(a2, b2, 1e-4, 1e-4)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, z), ["[1,108,400]", "[1,96,190]", "[1,36,120]"], "test_nn_Fold")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
