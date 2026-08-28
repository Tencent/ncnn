# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = x.new_ones((2,2))
        out1 = x.new_ones(3)
        out2 = x.new_ones((4,5,6,7,8))
        out3 = x.new_ones((1,2,1))
        out4 = x.new_ones((3,3,3,3))
        return out0, out1, out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_Tensor_new_ones",
        pnnx_args=("inputshape=[1,16]",),
    )

    b = mod.test_inference()

    # test shape only for uninitialized data
    passed = True
    for a0, b0 in zip(a, b):
        if not a0.shape == b0.shape:
            passed = False
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
