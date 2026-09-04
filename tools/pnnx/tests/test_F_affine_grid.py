# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.affine_grid(x, torch.Size((32, 3, 24, 24)), align_corners=False)

        y = F.affine_grid(y, torch.Size((12, 3, 10, 20, 30)), align_corners=False)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(32, 2, 3)
    y = torch.rand(12, 3, 4)

    a0, a1 = net(x, y)

    return test_model_formats(net, (x, y), (a0, a1), "test_F_affine_grid")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
