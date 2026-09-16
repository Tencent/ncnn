# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x.view(1, 1, 8).reshape(2, -1)
        y = y.reshape(-1, x.size(0)).unsqueeze(1)
        z = z.unsqueeze(0).unsqueeze(2).view(-1)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(8)
    y = torch.rand(9, 10)
    z = torch.rand(8, 9, 10)

    a = net(x, y, z)

    return test_model_formats(net, (x, y, z), a, "test_pnnx_fuse_adjacent_reshape")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
