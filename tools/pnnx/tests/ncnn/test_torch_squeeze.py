# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.squeeze(x, 0)
        y = torch.squeeze(y, 1)
        z = torch.squeeze(z)
        w = torch.squeeze(w, 2)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(3, 1)
    z = torch.rand(5, 1, 11)
    w = torch.rand(5, 9, 1, 33)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_squeeze.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_squeeze.pt inputshape=[1,16],[3,1],[5,1,11],[5,9,1,33]")

    # ncnn inference
    import test_torch_squeeze_ncnn
    b = test_torch_squeeze_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
