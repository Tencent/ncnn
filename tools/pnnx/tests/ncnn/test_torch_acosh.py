# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.acosh(x + 1)
        y = torch.acosh(y + 1)
        z = torch.acosh(z + 1)
        w = F.max_pool2d(w, 1)
        w = torch.acosh(w + 1)
        return x, y, z, w


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
    mod.save("test_torch_acosh.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_acosh.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_acosh_ncnn
    b = test_torch_acosh_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-4, 1e-4) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, y, z, w), ["[3,16]", "[5,9,11]", "[8,5,9,10]", "[2,3,5,7]"], "test_torch_acosh")

    return ts_ok and (pt2_ok is not False)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
