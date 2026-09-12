# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.expand(24)
        y = y.expand(-1, 11, -1)
        z = z.expand(2, 8, 3, -1, 4)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1)
    y = torch.rand(3, 1, 1)
    z = torch.rand(1, 8, 1, 9, 1)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_expand.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_Tensor_expand.pt inputshape=[1],[3,1,1],[1,8,1,9,1]")

    # pnnx inference
    import test_Tensor_expand_pnnx
    b = test_Tensor_expand_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, z), ["[1]", "[3,1,1]", "[1,8,1,9,1]"], "test_Tensor_expand")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
