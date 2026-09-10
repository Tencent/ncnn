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
        out0 = torch.index_select(x, 0, y)
        out1 = torch.index_select(x, 1, y)
        out2 = torch.index_select(x, 2, y)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(10, 13, 16)
    y = torch.randint(10, (8, ), dtype=torch.long)

    a = net(x, y)
    return test_model_formats(
        net,
        (x, y),
        a,
        "test_torch_index_select",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
