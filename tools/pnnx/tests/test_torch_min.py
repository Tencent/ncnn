# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x, x_indices = torch.min(x, dim=1, keepdim=False)
        y = torch.min(y)
        w = torch.min(z, w)
        z, z_indices = torch.min(z, dim=0, keepdim=True)
        return x, x_indices, y, z, z_indices, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    w = torch.rand(5, 9, 10)

    a = net(x, y, z, w)
    return test_model_formats(
        net,
        (x, y, z, w),
        a,
        "test_torch_min",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
