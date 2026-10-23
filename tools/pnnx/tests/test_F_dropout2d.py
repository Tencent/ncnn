# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.dropout2d(x, training=False)
        z = F.dropout2d(y, p=0.6, training=False)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 2, 16)
    y = torch.rand(1, 3, 12, 16)

    a0, a1 = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_F_dropout2d",
        pnnx_args=("inputshape=[1,12,2,16],[1,3,12,16]",),
    )

    b0, b1 = mod.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
