# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        return x + z[1], y[0] + y[1], y[1] - z[0] + z[1] - z[2]


class FlatInputModel(nn.Module):
    def __init__(self, model):
        super(FlatInputModel, self).__init__()
        self.model = model

    def forward(self, x, y0, y1, z0, z1, z2):
        return self.model(x, [y0, y1], [z0, z1, z2])

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4)
    y0 = torch.rand(2, 3, 4)
    y1 = torch.rand(2, 3, 4)
    z0 = torch.rand(2, 3, 4)
    z1 = torch.rand(2, 3, 4)
    z2 = torch.rand(2, 3, 4)
    y = [y0, y1]
    z = [z0, z1, z2]
    flat_net = FlatInputModel(net)
    flat_net.eval()
    flat_inputs = (x, y0, y1, z0, z1, z2)

    a = net(x, y, z)

    return test_model_formats(flat_net, flat_inputs, a, "test_pnnx_fuse_input_unpack")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
