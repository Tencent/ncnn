# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = torch.arange(x[0])
        out1 = torch.arange(x[1], 33, dtype=None)
        out2 = torch.arange(x[3], x[4], x[6] * 0.1, dtype=torch.float)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (16,), dtype=torch.int)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_torch_arange.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_arange.pt inputshape=[16]i32")

    # pnnx inference
    import test_torch_arange_pnnx
    b = test_torch_arange_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
