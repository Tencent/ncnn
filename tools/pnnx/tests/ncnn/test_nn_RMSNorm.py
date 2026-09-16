# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rmsn_0 = nn.RMSNorm(64)
        self.rmsn_0.weight = nn.Parameter(torch.rand(64))
        self.rmsn_1 = nn.RMSNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)
        self.rmsn_2 = nn.RMSNorm(normalized_shape=(3,24,64), eps=1e-3)

    def forward(self, x, y, w, q):
        x = self.rmsn_0(x)
        y = self.rmsn_0(y)
        z = self.rmsn_1(y)
        w0 = self.rmsn_0(w)
        w1 = self.rmsn_1(w)
        w2 = self.rmsn_2(w)
        q = self.rmsn_1(q)
        return x, y, z, w0, w1, w2, q

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 2, 3, 24, 64)
    q = torch.rand(2, 3, 24, 64)

    a = net(x, y, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, w, q))
    mod.save("test_nn_RMSNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_RMSNorm.pt inputshape=[1,24,64],[1,12,24,64],[1,2,3,24,64],[2,3,24,64]")

    # ncnn inference
    import test_nn_RMSNorm_ncnn
    b = test_nn_RMSNorm_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-3, 1e-3) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, y, w, q), ["[1,24,64]", "[1,12,24,64]", "[1,2,3,24,64]", "[2,3,24,64]"], "test_nn_RMSNorm")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
