# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0 = torch.unsqueeze(x, 0)
        x1 = torch.unsqueeze(x, 1)
        y0 = torch.unsqueeze(y, 1)
        y1 = torch.unsqueeze(y, -1)
        z0 = torch.unsqueeze(z, 0)
        z1 = torch.unsqueeze(z, -2)
        return x0, x1, y0, y1, z0, z1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(9, 11)
    z = torch.rand(4, 6, 7)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_unsqueeze.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_unsqueeze.pt inputshape=[16],[9,11],[4,6,7]")

    # ncnn inference
    import test_torch_unsqueeze_ncnn
    b = test_torch_unsqueeze_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
