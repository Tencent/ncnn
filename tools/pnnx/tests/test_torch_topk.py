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
        x, _ = torch.topk(x, 4)
        y, _ = torch.topk(y, k=1, dim=2, largest=False)
        z, indices = torch.topk(z, k=3, dim=-1, sorted=False)
        return x, y, z, indices

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
        "test_torch_topk",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
