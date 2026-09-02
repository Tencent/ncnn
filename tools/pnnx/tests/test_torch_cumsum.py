# Copyright 2021 Tencent
# Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        # x - 3d
        # y - 2d
        # z - 1d
        x0 = torch.cumsum(x, dim=0)
        x1 = torch.cumsum(x, dim=1)
        x2 = torch.cumsum(x, dim=2)

        y0 = torch.cumsum(y, dim=0)
        y1 = torch.cumsum(y, dim=1)

        z0 = torch.cumsum(z, dim=0)
        return x0, x1, x2, y0, y1, z0

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 16)
    y = torch.rand(5, 9)
    z = torch.rand(14)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_cumsum",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
