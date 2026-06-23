# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w):
        x = torch.flatten(x)
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        z = torch.flatten(z, start_dim=1, end_dim=2)
        w = self.conv(w)
        w0 = torch.flatten(w, start_dim=0)
        w1 = torch.flatten(w, start_dim=2)
        w2 = torch.flatten(w, start_dim=0, end_dim=1)
        return x, y, z, w0, w1, w2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_flatten.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_flatten.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_flatten_ncnn
    b = test_torch_flatten_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
