# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w0 = nn.Parameter(torch.rand(1, 12, 52))
        self.w1 = nn.Parameter(torch.rand(1, 12, 52))
        self.w2 = nn.Parameter(torch.rand(1, 12, 1))
        self.w3 = nn.Parameter(torch.rand(1, 12, 52))

    def forward(self, x):
        b = (self.w0 + self.w1 + 0.22) + self.w2 * 0.1
        x = x + b - self.w3 / 2
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 52)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_pnnx_fold_constant")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
