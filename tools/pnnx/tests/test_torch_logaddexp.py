# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.logaddexp(x, y)
        out1 = torch.logaddexp(y, y)
        out2 = torch.logaddexp(z, torch.ones_like(z) + 0.5)
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
        "test_torch_logaddexp",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
