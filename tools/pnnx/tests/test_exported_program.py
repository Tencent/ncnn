# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(3, 4, 1)
        self.register_buffer("scale", torch.rand(1, 4, 1, 1))
        self.register_buffer("offset", torch.rand(1, 4, 1, 1), persistent=False)
        self.constant = torch.rand(1, 4, 1, 1)

    def forward(self, x):
        x = self.conv(x) * self.scale + self.offset + self.constant
        return F.relu(x), torch.cat((x, x + 1), dim=1)


def test():
    if not has_torch_export():
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 4)
    a0, a1 = net(x)

    converted = exported_program_to_pnnx(net, x, "test_exported_program")
    b0, b1 = converted(x)

    return torch.equal(a0, b0) and torch.equal(a1, b1)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
