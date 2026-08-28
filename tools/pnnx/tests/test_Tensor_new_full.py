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
        out0 = x.new_full((2,2), 1.5)
        out1 = x.new_full((3,), 3)
        out2 = x.new_full((4,5,6,7,8), -0.5)
        out3 = x.new_full((1,2,1), 0)
        out4 = x.new_full((3,3,3,3), 1, dtype=torch.long)
        out5 = x.new_full((), 2.25)
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_Tensor_new_full",
        pnnx_args=("inputshape=[1,16]",),
    )

    b = mod.test_inference()

    passed = True
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape or a0.dtype != b0.dtype or not torch.equal(a0, b0):
            passed = False
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
