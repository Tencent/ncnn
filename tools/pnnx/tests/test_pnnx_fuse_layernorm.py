# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = LayerNorm2d(64)

    def forward(self, x):
        x = self.ln_0(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64, 16, 16)

    a0 = net(x)

    mod = convert_and_import(
        net,
        (x,),
        "test_pnnx_fuse_layernorm",
        pnnx_args=("inputshape=[1,64,16,16]",),
    )
    b0 = mod.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
