# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, s, t):
        out0 = torch.cat((x, y), dim=0)
        out1 = torch.cat((z, w), dim=2)
        out2 = torch.cat((w, w), dim=1)
        out3 = torch.cat((s, t), dim=0)
        out4 = torch.cat((s, s), dim=0)
        out5 = torch.cat((t, t), dim=0)
        out6 = torch.cat((t, t), dim=-3)
        return out0, out1, out2, out3, out4, out5, out6

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(2, 16)
    z = torch.rand(5, 9, 11)
    w = torch.rand(5, 9, 3)
    s = torch.rand(12, 3, 9, 3)
    t = torch.rand(2, 3, 9, 3)

    a = net(x, y, z, w, s, t)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, s, t))
    mod.save("test_torch_cat.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_cat.pt inputshape=[3,16],[2,16],[5,9,11],[5,9,3],[12,3,9,3],[2,3,9,3]")

    # ncnn inference
    import test_torch_cat_ncnn
    b = test_torch_cat_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
