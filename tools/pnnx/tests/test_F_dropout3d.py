# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.dropout3d(x, training=False)
        z = F.dropout3d(y, p=0.6, training=False)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 5, 2, 16)
    y = torch.rand(1, 3, 4, 12, 16)

    a0, a1 = net(x, y)

    return test_model_formats(net, (x, y), (a0, a1), "test_F_dropout3d")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
