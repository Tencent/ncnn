# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rmsn_0 = T5LayerNorm(26)
        self.rmsn_1 = T5LayerNorm(21)

    def forward(self, x, y):
        x = self.rmsn_0(x)
        y = self.rmsn_1(y)
        return x, y

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64, 26)
    y = torch.rand(3, 15, 15, 21)

    a0, a1 = net(x, y)

    # export onnx
    torch.onnx.export(net, (x,y), "test.onnx")

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_pnnx_fuse_rmsnorm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_rmsnorm.pt inputshape=[1,64,26],[3,15,15,21]")

    # pnnx inference
    import test_pnnx_fuse_rmsnorm_pnnx
    b0, b1 = test_pnnx_fuse_rmsnorm_pnnx.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
