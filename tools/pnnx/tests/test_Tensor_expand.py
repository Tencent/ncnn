# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.expand(24)
        y = y.expand(-1, 11, -1)
        z = z.expand(2, 8, 3, -1, 4)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1)
    y = torch.rand(3, 1, 1)
    z = torch.rand(1, 8, 1, 9, 1)

    a = net(x, y, z)

    return test_model_formats(net, (x, y, z), a, "test_Tensor_expand")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
