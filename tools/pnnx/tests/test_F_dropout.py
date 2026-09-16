# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = F.dropout(x, training=False)
        y = F.dropout(y, training=False)
        z = F.dropout(z, p=0.6, training=False)
        w = F.dropout(w, p=0.1, training=False)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a0, a1, a2, a3 = net(x, y, z, w)

    return test_model_formats(net, (x, y, z, w), (a0, a1, a2, a3), "test_F_dropout")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
