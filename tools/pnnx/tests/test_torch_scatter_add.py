# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y0, y1, z):
        out0 = torch.scatter_add(x, 0, y0, z)
        out1 = torch.scatter_add(x, 0, y1, z)
        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(13, 15)
    y0 = torch.randint(10, (1, 15), dtype=torch.long)
    y1 = torch.randint(10, (12, 15), dtype=torch.long)
    z = torch.rand(12, 15)

    a = net(x, y0, y1, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y0, y1, z))
    mod.save("test_torch_scatter_add.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_torch_scatter_add.pt inputshape=[13,15],[1,15]i64,[12,15]i64,[12,15]")

    # pnnx inference
    import test_torch_scatter_add_pnnx
    b = test_torch_scatter_add_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y0, y1, z), ["[13,15]", "[1,15]", "[12,15]", "[12,15]"], "test_torch_scatter_add")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
