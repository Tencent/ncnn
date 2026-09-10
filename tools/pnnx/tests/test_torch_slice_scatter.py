# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        if version.parse(torch.__version__) >= version.parse('1.13') and version.parse(torch.__version__) < version.parse('2.0'):
            out0 = torch.slice_scatter(x, y, start=6, step=1)
        else:
            out0 = torch.slice_scatter(x, y, start=6)
        out1 = torch.slice_scatter(x, z, dim=1, start=2, end=6, step=2)
        return out0, out1

def test():
    if version.parse(torch.__version__) < version.parse('1.11'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(8, 8)
    y = torch.rand(2, 8)
    z = torch.rand(8, 2)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_slice_scatter",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
