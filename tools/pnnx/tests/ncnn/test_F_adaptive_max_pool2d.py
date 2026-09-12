# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, q):
        out0 = F.adaptive_max_pool2d(x, output_size=(7,6))
        out1 = F.adaptive_max_pool2d(x, output_size=1)
        out2 = F.adaptive_max_pool2d(x, output_size=(None,3))
        out3 = F.adaptive_max_pool2d(x, output_size=(5,None))
        q = F.adaptive_max_pool2d(q, output_size=(7,6))
        return out0, out1, out2, out3, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    q = torch.rand(2, 12, 24, 64)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_F_adaptive_max_pool2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_max_pool2d.pt inputshape=[1,12,24,64],[2,12,24,64]")

    # ncnn inference
    import test_F_adaptive_max_pool2d_ncnn
    b = test_F_adaptive_max_pool2d_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-4, 1e-4) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, q), ["[1,12,24,64]", "[2,12,24,64]"], "test_F_adaptive_max_pool2d")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
