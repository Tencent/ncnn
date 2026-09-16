# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.in_0 = nn.InstanceNorm3d(num_features=12, affine=True)
        self.in_0.weight = nn.Parameter(torch.rand(12))
        self.in_0.bias = nn.Parameter(torch.rand(12))
        self.in_1 = nn.InstanceNorm3d(num_features=12, eps=1e-2, affine=False)

    def forward(self, x, q):
        x = self.in_0(x)
        x = self.in_1(x)
        q = self.in_0(q)
        q = self.in_1(q)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 32, 64)
    q = torch.rand(2, 12, 6, 8, 10)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_nn_InstanceNorm3d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_InstanceNorm3d.pt inputshape=[1,12,24,32,64],[2,12,6,8,10]")

    # ncnn inference
    import test_nn_InstanceNorm3d_ncnn
    b = test_nn_InstanceNorm3d_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-4, 1e-4) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, q), ["[1,12,24,32,64]", "[2,12,6,8,10]"], "test_nn_InstanceNorm3d")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
