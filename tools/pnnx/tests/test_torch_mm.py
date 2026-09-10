# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a0, a1):
        a = torch.mm(a0, a1)
        return a

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(23, 14)
    a1 = torch.rand(14, 35)

    a = net(a0, a1)
    return test_model_formats(
        net,
        (a0, a1),
        a,
        "test_torch_mm",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
