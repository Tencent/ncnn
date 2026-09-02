# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.upsample_bilinear(x, size=(12,12))
        x = F.upsample_bilinear(x, scale_factor=2)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_F_upsample_bilinear")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
