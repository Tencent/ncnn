# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w3 = nn.Parameter(torch.rand(24))
        self.b3 = nn.Parameter(torch.rand(24))
        self.w4 = nn.Parameter(torch.rand(12, 16))
        self.b4 = nn.Parameter(torch.rand(12, 16))
        self.w5 = nn.Parameter(torch.rand(24))
        self.b5 = nn.Parameter(torch.rand(24))

    def forward(self, x, y, z, w0, b0, w1, b1, w2, b2):
        x = F.layer_norm(x, (24,), w0, b0)
        x = F.layer_norm(x, (12,24), None, None)
        x = F.layer_norm(x, (24,), self.w3, self.b3)

        y = F.layer_norm(y, (16,), None, None, eps=1e-3)
        y = F.layer_norm(y, (12,16), w1, b1)
        y = F.layer_norm(y, (12,16), self.w4, self.b4)

        z = F.layer_norm(z, (24,), w2, b2)
        z = F.layer_norm(z, (12,16,24), None, None, eps=1e-2)
        z = F.layer_norm(z, (24,), self.w5, self.b5)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)
    w0 = torch.rand(24)
    b0 = torch.rand(24)
    w1 = torch.rand(12, 16)
    b1 = torch.rand(12, 16)
    w2 = torch.rand(24)
    b2 = torch.rand(24)

    a0, a1, a2 = net(x, y, z, w0, b0, w1, b1, w2, b2)

    inputs = (x, y, z, w0, b0, w1, b1, w2, b2)
    return test_model_formats(net, inputs, (a0, a1, a2), "test_F_layer_norm")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
