# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.Dropout3d()
        self.dropout_1 = nn.Dropout3d(p=0.7)

    def forward(self, x, y):
        x = self.dropout_0(x)
        y = self.dropout_1(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 6, 8, 16)
    y = torch.rand(1, 3, 4, 5, 6)

    a0, a1 = net(x, y)

    return test_model_formats(net, (x, y), (a0, a1), "test_nn_Dropout3d")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
