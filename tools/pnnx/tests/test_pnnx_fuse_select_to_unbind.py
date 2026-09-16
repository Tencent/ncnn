# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x0 = torch.select(x, 0, 0)
        x1 = torch.select(x, 0, 1)
        x2 = torch.select(x, 0, 2)

        z4 = torch.select(x, 2, 4)
        z3 = torch.select(x, 2, 3)

        y0 = torch.select(x, 1, 0)
        y1 = torch.select(x, 1, 1)
        y2 = torch.select(x, 1, 2)
        y3 = torch.select(x, 1, 3)

        z2 = torch.select(x, 2, 2)
        z1 = torch.select(x, 2, 1)
        z0 = torch.select(x, 2, 0)

        return x0, x1, x2, y0, y1, y2, y3, z0, z1, z2, z3, z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_pnnx_fuse_select_to_unbind")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
