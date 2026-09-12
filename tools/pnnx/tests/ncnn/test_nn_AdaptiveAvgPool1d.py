# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveAvgPool1d(output_size=(7))
        self.pool_1 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x, q):
        x = self.pool_0(x)
        x = self.pool_1(x)
        q = self.pool_0(q)
        q = self.pool_1(q)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13)
    q = torch.rand(2, 128, 13)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_nn_AdaptiveAvgPool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_AdaptiveAvgPool1d.pt inputshape=[1,128,13],[2,128,13]")

    # ncnn inference
    import test_nn_AdaptiveAvgPool1d_ncnn
    b = test_nn_AdaptiveAvgPool1d_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-4, 1e-4) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, q), ["[1,128,13]", "[2,128,13]"], "test_nn_AdaptiveAvgPool1d")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
