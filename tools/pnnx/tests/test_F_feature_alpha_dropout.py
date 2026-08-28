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
        x = F.feature_alpha_dropout(x, training=False)
        y = F.feature_alpha_dropout(y, p=0.6, training=False)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 12, 16)
    y = torch.rand(1, 5, 7, 9, 11)

    a0, a1 = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_F_feature_alpha_dropout",
        pnnx_args=("inputshape=[1,3,4,12,16],[1,5,7,9,11]",),
    )

    b0, b1 = mod.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
