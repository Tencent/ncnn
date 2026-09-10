# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

def silu_forward_0(x):
    return x * torch.sigmoid(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1
        x = F.silu(x)
        y = F.silu(y)
        z = F.silu(z)
        w = silu_forward_0(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a = net(x, y, z, w)

    compare = lambda x, y: torch.allclose(x, y, 1e-4, 1e-4)
    return test_model_formats(net, (x, y, z, w), a, "test_F_silu", compare)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
