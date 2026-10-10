# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x, output_size=7)
        x = F.adaptive_avg_pool1d(x, output_size=1)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_F_adaptive_avg_pool1d",
        pnnx_args=("inputshape=[1,12,24]",),
    )

    b = mod.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
