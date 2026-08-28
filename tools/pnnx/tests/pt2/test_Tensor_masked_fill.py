# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        mask = x > y
        out = x.masked_fill(mask, 2)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)

    a = net(x, y)

    # export pt2
    mod = torch.export.export(net, (x, y))
    torch.export.save(mod, "test_Tensor_masked_fill.pt2")

    # pt2 to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_masked_fill.pt2 inputshape=[6,16],[6,16]")

    # pnnx inference
    import test_Tensor_masked_fill_pnnx
    b = test_Tensor_masked_fill_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
