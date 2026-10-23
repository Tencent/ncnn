# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveAvgPool2d(output_size=(7,6))
        self.pool_1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.pool_2 = nn.AdaptiveAvgPool2d(output_size=(None,3))
        self.pool_3 = nn.AdaptiveAvgPool2d(output_size=(5,None))

    def forward(self, x):
        out0 = self.pool_0(x)
        out1 = self.pool_1(x)
        out2 = self.pool_2(x)
        out3 = self.pool_3(x)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13, 13)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_nn_AdaptiveAvgPool2d",
        pnnx_args=("inputshape=[1,128,13,13]",),
    )

    b = mod.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
