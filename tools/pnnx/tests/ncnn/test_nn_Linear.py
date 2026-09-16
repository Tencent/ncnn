# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.linear_1 = nn.Linear(in_features=16, out_features=13, bias=True)

        self.linear_2 = nn.Linear(in_features=13, out_features=17, bias=True)
        if version.parse(torch.__version__) < version.parse('1.9'):
            # weight_norm on torch 1.8 produces wrong output shape, skip it
            pass
        elif version.parse(torch.__version__) < version.parse('2.1'):
            self.linear_2 = torch.nn.utils.weight_norm(self.linear_2)
        else:
            self.linear_2 = torch.nn.utils.parametrizations.weight_norm(self.linear_2)

    def forward(self, x, y, z, w, q, r):
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.linear_2(x)

        y = self.linear_0(y)
        y = self.linear_1(y)
        y = self.linear_2(y)

        z = self.linear_0(z)
        z = self.linear_1(z)
        z = self.linear_2(z)
        z = F.relu(z)

        w = self.linear_0(w)
        w = self.linear_1(w)
        w = self.linear_2(w)
        q = F.max_pool2d(q, 1)
        q = self.linear_0(q)
        q = self.linear_1(q)
        q = self.linear_2(q)
        # rank-3 input with a non-singleton middle dim (regression: ncnn Gemm
        # N1M layout would drop the h dim, so pnnx must flatten it instead)
        r = self.linear_0(r)
        r = self.linear_1(r)
        r = self.linear_2(r)
        return x, y, z, w, q, r

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(64)
    y = torch.rand(12, 64)
    z = torch.rand(1, 3, 12, 64)
    w = torch.rand(1, 64)
    q = torch.rand(2, 3, 5, 64)
    r = torch.rand(4, 7, 64)

    a = net(x, y, z, w, q, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q, r))
    mod.save("test_nn_Linear.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Linear.pt inputshape=[64],[12,64],[1,3,12,64],[1,64],[2,3,5,64],[4,7,64]")

    # ncnn inference
    import test_nn_Linear_ncnn
    b = test_nn_Linear_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-3, 1e-3) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, y, z, w, q, r), ["[64]", "[12,64]", "[1,3,12,64]", "[1,64]", "[2,3,5,64]", "[4,7,64]"], "test_nn_Linear")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
