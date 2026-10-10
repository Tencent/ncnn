# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q, r, s):
        out0 = torch.atan2(x, y)
        out1 = torch.atan2(y, y)
        out2 = torch.atan2(z, torch.ones_like(z) + 0.5)
        q = F.max_pool2d(q, 1)
        r = F.max_pool2d(r, 1)
        out3 = torch.atan2(q, r)
        out4 = torch.atan2(q, s)
        return out0, out1, out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)
    q = torch.rand(2, 3, 5, 7)
    r = torch.rand(2, 3, 5, 7)
    s = torch.rand(3, 5, 7)

    a = net(x, y, z, q, r, s)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q, r, s))
    mod.save("test_torch_atan2.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_atan2.pt inputshape=[3,16],[3,16],[5,9,3],[2,3,5,7],[2,3,5,7],[3,5,7]")

    # ncnn inference
    import test_torch_atan2_ncnn
    b = test_torch_atan2_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
