# Copyright 2021 Tencent
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
        z = torch.flatten(z, start_dim=3, end_dim=4)
        w = self.conv(w)
        w0 = torch.flatten(w, start_dim=0)
        w1 = torch.flatten(w, start_dim=2)
        w2 = torch.flatten(w, start_dim=0, end_dim=1)
        return x, y, z, w0, w1, w2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_flatten.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_flatten.pt inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10],[2,3,5,7]")

    # pnnx inference
    import test_torch_flatten_pnnx
    b = test_torch_flatten_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
