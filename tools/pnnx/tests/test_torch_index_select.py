# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = torch.index_select(x, 0, y)
        out1 = torch.index_select(x, 1, y)
        out2 = torch.index_select(x, 2, y)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(10, 13, 16)
    y = torch.randint(10, (8, ), dtype=torch.long)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_index_select.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_index_select.pt inputshape=[10,13,16],[8]i64")

    # pnnx inference
    import test_torch_index_select_pnnx
    b = test_torch_index_select_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
