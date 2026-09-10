# Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0 = F.glu(x, dim=0)

        y0 = F.glu(y, dim=0)
        y1 = F.glu(y, dim=1)

        z0 = F.glu(z, dim=0)
        z1 = F.glu(z, dim=1)
        z2 = F.glu(z, dim=2)
        return x0, y0, y1, z0, z1, z2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18)
    y = torch.rand(12, 16)
    z = torch.rand(24, 28, 34)

    x0, y0, y1, z0, z1, z2 = net(x, y, z)

    outputs = (x0, y0, y1, z0, z1, z2)
    return test_model_formats(net, (x, y, z), outputs, "test_F_glu")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
