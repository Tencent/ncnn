# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = x + 3
        out1 = x - 4j
        out2 = x * (1.2-0.9j - out0)
        return out0, out1, out2

def test():
    if version.parse(torch.__version__) < version.parse('1.9'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 15, dtype=torch.complex64)

    a = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_ir_complex",
        pnnx_args=("inputshape=[3,15]c64",),
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
