# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x[:2,:].fill_(z[0])
        y[:1,:].fill_(0.22)
        return x + y.fill_(7)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)
    z = torch.rand(1)

    a = net(x, y, z)

    # This model mutates views of caller-owned x/y, not local temporaries.
    # PT2 inference import must not silently erase those external writes.
    return test_model_formats(
        net, (x, y, z), a, "test_Tensor_fill",
        unsupported_by_pnnx_pt2="unsupported alias write/mutation of argument self",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
