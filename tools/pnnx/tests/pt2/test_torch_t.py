# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = torch.t(x)
        y = torch.t(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3)
    y = torch.rand(5, 9)

    a = net(x, y)

    # export pt2
    mod = torch.export.export(net, (x, y))
    torch.export.save(mod, "test_torch_t.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_torch_t.pt2 inputshape=[3],[5,9]")

    # pnnx inference
    import test_torch_t_pnnx
    b = test_torch_t_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
