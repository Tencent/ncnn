# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.in_0 = nn.InstanceNorm1d(num_features=12, affine=True)
        self.in_0.weight = nn.Parameter(torch.rand(12))
        self.in_0.bias = nn.Parameter(torch.rand(12))
        self.in_1 = nn.InstanceNorm1d(num_features=12, eps=1e-2, affine=False)
        self.in_2 = nn.InstanceNorm1d(num_features=12, eps=1e-4, affine=True, track_running_stats=True)
        self.in_2.weight = nn.Parameter(torch.rand(12))
        self.in_2.bias = nn.Parameter(torch.rand(12))

    def forward(self, x):
        x = self.in_0(x)
        x = self.in_1(x)
        x = self.in_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_InstanceNorm1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_InstanceNorm1d.pt inputshape=[1,12,24]")

    # pnnx inference
    import test_nn_InstanceNorm1d_pnnx
    b = test_nn_InstanceNorm1d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
