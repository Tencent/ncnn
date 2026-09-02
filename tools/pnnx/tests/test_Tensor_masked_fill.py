# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        mask = x > y
        out = x.masked_fill(mask, 2)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)

    a = net(x, y)

    return test_model_formats(net, (x, y), a, "test_Tensor_masked_fill")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
