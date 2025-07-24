# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x, x_indices = torch.min(x, dim=1, keepdim=False)
        y = torch.min(y)
        w = torch.min(z, w)
        z, z_indices = torch.min(z, dim=0, keepdim=True)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(5, 9, 10)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_min.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_min.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[5,9,10]")

    # ncnn inference
    import test_torch_min_ncnn
    b = test_torch_min_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
