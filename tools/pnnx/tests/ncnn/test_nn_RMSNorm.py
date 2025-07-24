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

    def forward(self, x, y):
        x = self.rmsn_0(x)
        y = self.rmsn_0(y)
        z = self.rmsn_1(y)
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_RMSNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_RMSNorm.pt inputshape=[1,24,64],[1,12,24,64]")

    # ncnn inference
    import test_nn_RMSNorm_ncnn
    b = test_nn_RMSNorm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
