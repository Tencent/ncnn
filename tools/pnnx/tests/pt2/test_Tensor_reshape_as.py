# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)

    a = net(x, y, z)

    program = torch.export.export(net, (x, y, z))
    torch.export.save(program, "test_Tensor_reshape_as.pt2")

    from pnnxutil import run_pnnx
    run_pnnx("test_Tensor_reshape_as.pt2", "inputshape=[1,3,16],[6,2,2,2],[48] inputshape2=[1,24,8],[12,1,4,4],[192]")

    # pnnx inference
    import test_Tensor_reshape_as_pnnx
    netb = test_Tensor_reshape_as_pnnx.Model().float().eval()
    b = netb(x, y, z)

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
