# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        c0 = torch.le(x, y)
        c1 = torch.ge(y, y)
        out0 = torch.where(c0, x + 2, x + 4)
        out1 = torch.where(c1, x + y, x - y)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)
    return test_model_formats(
        net,
        (x, y),
        a,
        "test_torch_where",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
