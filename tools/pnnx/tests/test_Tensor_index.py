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
        x = x[[2, 1], [0, 1]]
        y = y[..., [1, 0]]
        z = z[[True, False], [False, False, True, False]]
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 6)
    y = torch.rand(5, 9, 2)
    z = torch.rand(2, 4, 5, 10)

    a = net(x, y, z)

    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_Tensor_index",
        unsupported_by_torch_export="Pending unbacked symbols",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
