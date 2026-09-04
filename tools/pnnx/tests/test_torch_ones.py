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
        x = torch.ones(x.size())
        y = torch.ones(y.size())
        z = torch.ones(z.size(), dtype=torch.long)
        return x, y, z

class InputAnchoredModel(nn.Module):
    def __init__(self, model):
        super(InputAnchoredModel, self).__init__()
        self.model = model

    def forward(self, x, y, z):
        out = self.model(x, y, z)
        zx = x.sum() * 0
        zy = y.sum() * 0
        zz = z.sum() * 0
        return out[0] + zx, out[1] + zy, out[2] + zz.to(dtype=out[2].dtype)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)
    wrapped = InputAnchoredModel(net)
    wrapped.eval()

    return test_model_formats(
        wrapped,
        (x, y, z),
        a,
        "test_torch_ones",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
