# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x * 10
        y = y * 13
        y = y.to(dtype=x.dtype, memory_format=torch.contiguous_format)
        x = x.to(device='cpu', dtype=torch.int, copy=True)
        x = x + 1
        y = y - 2
        z = x.to(y.device)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.randint(10, (1, 13), dtype=torch.int)

    a = net(x, y)

    # export pt2
    mod = torch.export.export(net, (x, y))
    torch.export.save(mod, "test_Tensor_to.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_Tensor_to.pt2 inputshape=[3,16],[1,13]i32")

    # pnnx inference
    import test_Tensor_to_pnnx
    b = test_Tensor_to_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
