# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, q):
        x = F.pixel_unshuffle(x, 4)
        x = F.pixel_unshuffle(x, 2)
        q = F.pixel_unshuffle(q, 4)
        q = F.pixel_unshuffle(q, 2)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)
    q = torch.rand(2, 3, 128, 128)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_F_pixel_unshuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_pixel_unshuffle.pt inputshape=[1,3,128,128],[2,3,128,128]")

    # ncnn inference
    import test_F_pixel_unshuffle_ncnn
    b = test_F_pixel_unshuffle_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-4, 1e-4) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, q), ["[1,3,128,128]", "[2,3,128,128]"], "test_F_pixel_unshuffle")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
