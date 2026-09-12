# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, torchscript_to_pnnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bn_0 = nn.BatchNorm2d(num_features=32)
        self.bn_1 = nn.BatchNorm2d(num_features=32, eps=1e-1, affine=False)
        self.bn_2 = nn.BatchNorm2d(num_features=11, affine=True)

    def forward(self, x, y):
        x = self.bn_0(x)
        x = self.bn_1(x)

        y = self.bn_2(y)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 32, 12, 64)
    y = torch.rand(1, 11, 1, 1)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_BatchNorm2d.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_nn_BatchNorm2d", "[1,32,12,64],[1,11,1,1]")

    # pnnx inference
    b0, b1 = converted(x, y)

    if not (torch.equal(a0, b0) and torch.equal(a1, b1)):
        return False

    if not has_torch_export():
        return True

    converted = exported_program_to_pnnx(net, (x, y), "test_nn_BatchNorm2d_pt2")
    c0, c1 = converted(x, y)

    return torch.equal(a0, c0) and torch.equal(a1, c1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
