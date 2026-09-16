# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.view(1, 1, 8).reshape(2, -1)
        y = y.reshape(-1, x.size(0)).unsqueeze(1)
        z = z.unsqueeze(0).unsqueeze(2).view(-1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(8)
    y = torch.rand(9, 10)
    z = torch.rand(8, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_pnnx_fuse_adjacent_reshape.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_pnnx_fuse_adjacent_reshape.pt inputshape=[8],[9,10],[8,9,10]")

    # pnnx inference
    import test_pnnx_fuse_adjacent_reshape_pnnx
    b = test_pnnx_fuse_adjacent_reshape_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, z), ["[8]", "[9,10]", "[8,9,10]"], "test_pnnx_fuse_adjacent_reshape")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
