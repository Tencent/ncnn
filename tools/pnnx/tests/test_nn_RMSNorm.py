# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rmsn_0 = nn.RMSNorm(64)
        self.rmsn_0.weight = nn.Parameter(torch.rand(64))
        self.rmsn_1 = nn.RMSNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)

    def forward(self, x, y, z):
        x = self.rmsn_0(x)
        x = self.rmsn_1(x)

        y = self.rmsn_0(y)
        y = self.rmsn_1(y)

        z = self.rmsn_0(z)
        z = self.rmsn_1(z)
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2 = net(x, y, z)

    return test_model_formats(net, (x, y, z), (a0, a1, a2), "test_nn_RMSNorm")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
