# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x.repeat(2, 3)
        x = x.repeat(3, 4)
        y = y.repeat(2, 1, 4)
        y = y.repeat(4, 5, 1)
        z = z.repeat(2, 3, 1, 5)
        z = z.repeat(3, 3, 1, 1)
        w = F.max_pool2d(w, 1)
        w0 = w.repeat(1, 2, 1, 1)
        w1 = w.repeat(1, 1, 1, 4)
        return x, y, z, w0, w1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 1)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_Tensor_repeat.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_repeat.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,1]")

    # ncnn inference
    import test_Tensor_repeat_ncnn
    b = test_Tensor_repeat_ncnn.test_inference()

    ts_ok = all(torch.equal(a0, b0) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, y, z, w), ["[3,16]", "[5,9,11]", "[8,5,9,10]", "[2,3,5,1]"], "test_Tensor_repeat")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
