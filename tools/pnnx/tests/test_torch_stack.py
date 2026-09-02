# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.stack((x, y), dim=0)
        out1 = torch.stack((x, y), dim=2)
        out2 = torch.stack((z, w), dim=2)
        out3 = torch.stack((z, w), dim=-1)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)
    w = torch.rand(5, 9, 3)

    a = net(x, y, z, w)
    return test_model_formats(
        net,
        (x, y, z, w),
        a,
        "test_torch_stack",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
