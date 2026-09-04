# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveAvgPool1d(output_size=(7))
        self.pool_1 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = self.pool_0(x)
        x = self.pool_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_nn_AdaptiveAvgPool1d")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
