# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0, x1, x2 = torch.unbind(x, dim=1)
        y0, y1, y2, y3, y4, y5, y6, y7, y8 = torch.unbind(y, dim=2)
        z0, z1, z2, z3 = torch.unbind(z, dim=0)
        return x0, x1, x2, y0, y1, y2, y3, y4, y5, y6, y7, y8, z0, z1, z2, z3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(4, 8, 5, 9, 10)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_unbind",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
