# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.cross(x, y)
        out1 = torch.cross(x, y, dim=1)
        out2 = torch.cross(z, w)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 3)
    y = torch.rand(3, 3)
    z = torch.rand(5, 3)
    w = torch.rand(5, 3)

    a = net(x, y, z, w)
    return test_model_formats(
        net,
        (x, y, z, w),
        a,
        "test_torch_cross",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
