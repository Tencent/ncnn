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
        out0 = torch.lt(x, y)
        out1 = torch.lt(y, y)
        out2 = torch.lt(z, 1)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_lt",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
