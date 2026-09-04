# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.positive(x)
        y = torch.positive(y)
        z = torch.positive(z)
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('1.9'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_positive",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
