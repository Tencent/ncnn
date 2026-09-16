# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        a = torch.ge(x, y)
        out = torch.bitwise_not(a)
        return out

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
        "test_torch_bitwise_not",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
