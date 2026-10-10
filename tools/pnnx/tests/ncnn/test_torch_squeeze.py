# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w, q, r):
        x = torch.squeeze(x, 0)
        y = torch.squeeze(y, 1)
        z = torch.squeeze(z)
        w = torch.squeeze(w, 2)
        q = self.conv(q)
        q0 = torch.squeeze(q)
        q1 = torch.squeeze(q, 2)
        q2 = torch.squeeze(q, 0)
        r = self.conv(r)
        r = torch.squeeze(r, 0)
        return x, y, z, w, q0, q1, q2, r

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(3, 1)
    z = torch.rand(5, 1, 11)
    w = torch.rand(5, 9, 1, 33)
    q = torch.rand(2, 3, 1, 5)
    r = torch.rand(1, 3, 1, 5)

    a = net(x, y, z, w, q, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q, r))
    mod.save("test_torch_squeeze.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_squeeze.pt inputshape=[1,16],[3,1],[5,1,11],[5,9,1,33],[2,3,1,5],[1,3,1,5]")

    # ncnn inference
    import test_torch_squeeze_ncnn
    b = test_torch_squeeze_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
