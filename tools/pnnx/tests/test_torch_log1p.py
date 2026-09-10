# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from pnnx_test_utils import test_model_formats


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.log1p(x)
        y = torch.log1p(y)
        z = torch.log1p(z)
        return x, y, z


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_log1p",
        compare=lambda a0, b0: torch.allclose(a0, b0, 1e-6, 1e-6),
    )


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
