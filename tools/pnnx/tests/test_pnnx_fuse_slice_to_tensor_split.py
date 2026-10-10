# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0 = x[:3]
        x1 = x[3:]

        z3 = z[:,:,7:z.size(2)]
        z2 = z[:,:,4:7]

        y0 = y[:2,:]
        y1 = y[2:4,:]
        y2 = y[4:,:]

        z1 = z[:,:,2:4]
        z0 = z[:,:,:2]

        return x0, x1, y0, y1, y2, z0, z1, z2, z3

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
    mod.save("test_pnnx_fuse_slice_to_tensor_split.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_slice_to_tensor_split.pt inputshape=[8],[9,10],[8,9,10]")

    # pnnx inference
    import test_pnnx_fuse_slice_to_tensor_split_pnnx
    b = test_pnnx_fuse_slice_to_tensor_split_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
