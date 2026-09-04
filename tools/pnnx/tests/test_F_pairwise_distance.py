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
        z1 = F.pairwise_distance(x,y,p=1,keepdim=False)
        z2 = F.pairwise_distance(x,y,p=2,keepdim=True)
        z3 = F.pairwise_distance(x,y)
        z4 = F.pairwise_distance(x,y,eps = 1e-3)
        return z1,z2,z3,z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 128, 128)
    y = torch.rand(12, 128, 128)

    a0,a1,a2,a3 = net(x, y)

    return test_model_formats(net, (x, y), (a0, a1, a2, a3), "test_F_pairwise_distance")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
