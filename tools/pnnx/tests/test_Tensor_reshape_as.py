# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)

    a = net(x, y, z)

    return test_model_formats(net, (x, y, z), a, "test_Tensor_reshape_as")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
