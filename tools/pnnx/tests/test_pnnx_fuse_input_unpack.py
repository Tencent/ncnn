# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        return x + z[1], y[0] + y[1], y[1] - z[0] + z[1] - z[2]

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4)
    y0 = torch.rand(2, 3, 4)
    y1 = torch.rand(2, 3, 4)
    z0 = torch.rand(2, 3, 4)
    z1 = torch.rand(2, 3, 4)
    z2 = torch.rand(2, 3, 4)
    y = [y0, y1]
    z = [z0, z1, z2]

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_pnnx_pnnx_fuse_input_unpack.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_pnnx_fuse_input_unpack.pt inputshape=[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4]")

    # pnnx inference
    import test_pnnx_pnnx_fuse_input_unpack_pnnx
    b = test_pnnx_pnnx_fuse_input_unpack_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
