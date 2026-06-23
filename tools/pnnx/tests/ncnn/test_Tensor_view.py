# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, v):
        x = x.reshape(2, 24)
        x = x.reshape(48)
        y = y.reshape(11, 5, 9)
        y = y.reshape(99, 5)
        z = z.reshape(4, 3, 6, 10)
        z = z.reshape(15, 6, 8)
        w = F.max_pool2d(w, 1)
        w0 = w.view(2, 3, 35)
        w1 = w.view(6, 5, 7)
        v = v.view(4, 2, 5, 7)
        v = v.permute(1, 0, 2, 3)
        v = F.max_pool2d(v, 1)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w0 = F.relu(w0)
        w1 = F.relu(w1)
        return x, y, z, w0, w1, v

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 2)
    w = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)

    a = net(x, y, z, w, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, v))
    mod.save("test_Tensor_view.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_view.pt inputshape=[3,16],[5,9,11],[8,5,9,2],[2,3,5,7],[280]")

    # ncnn inference
    import test_Tensor_view_ncnn
    b = test_Tensor_view_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
