# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.glu0 = nn.GLU(dim=0)
        self.glu1 = nn.GLU(dim=1)
        self.glu2 = nn.GLU(dim=2)

    def forward(self, x, y, z):
        x0 = self.glu0(x)

        y0 = self.glu0(y)
        y1 = self.glu1(y)

        z0 = self.glu0(z)
        z1 = self.glu1(z)
        z2 = self.glu2(z)
        return x0, y0, y1, z0, z1, z2


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18)
    y = torch.rand(12, 16)
    z = torch.rand(24, 28, 34)

    x0, y0, y1, z0, z1, z2 = net(x, y, z)

    return test_model_formats(net, (x, y, z), (x0, y0, y1, z0, z1, z2), "test_nn_GLU")


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
