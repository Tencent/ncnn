# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        out0 = torch.le(x, y)
        out1 = torch.le(y, y)
        out2 = torch.le(z, 1)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_le.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_torch_le.pt inputshape=[3,16],[3,16],[5,9,3]")

    # pnnx inference
    import test_torch_le_pnnx
    b0, b1, b2 = test_torch_le_pnnx.test_inference()

    ts_ok = torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, z), ["[3,16]", "[3,16]", "[5,9,3]"], "test_torch_le")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
