# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.Dropout()
        self.dropout_1 = nn.Dropout(p=0.7)

    def forward(self, x, y, z, w):
        x = self.dropout_0(x)
        y = self.dropout_0(y)
        z = self.dropout_1(z)
        w = self.dropout_1(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 12, 24, 32, 64)

    a0, a1, a2, a3 = net(x, y, z, w)

    return test_model_formats(net, (x, y, z, w), (a0, a1, a2, a3), "test_nn_Dropout")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
