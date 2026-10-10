# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out = torch.bitwise_right_shift(x, y)
        return out

def test():
    if version.parse(torch.__version__) < version.parse('1.10'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (3, 16), dtype=torch.int)
    y = torch.randint(10, (3, 16), dtype=torch.int)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_bitwise_right_shift.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_bitwise_right_shift.pt inputshape=[3,16]i32,[3,16]i32")

    # pnnx inference
    import test_torch_bitwise_right_shift_pnnx
    b = test_torch_bitwise_right_shift_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
