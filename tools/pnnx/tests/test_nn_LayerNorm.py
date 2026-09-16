# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LayerNorm(64)
        self.ln_0.weight = nn.Parameter(torch.rand(64))
        self.ln_0.bias = nn.Parameter(torch.rand(64))
        self.ln_1 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)

    def forward(self, x, y, z):
        x = self.ln_0(x)
        x = self.ln_1(x)

        y = self.ln_0(y)
        y = self.ln_1(y)

        z = self.ln_0(z)
        z = self.ln_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2 = net(x, y, z)

    return test_model_formats(net, (x, y, z), (a0, a1, a2), "test_nn_LayerNorm")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
