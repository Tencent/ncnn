# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pnnx

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x

if __name__ == "__main__":
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a0 = net(x)

    mod = torch.jit.trace(net, x)
    mod.save("test_F_relu.pt")

    pnnx.convert("test_F_relu.pt", [1, 16], "f32")
