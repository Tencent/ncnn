# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x * 10
        y = y * 13
        y = y.to(dtype=x.dtype, memory_format=torch.contiguous_format)
        x = x.to(device='cpu', dtype=torch.int, copy=True)
        x = x + 1
        y = y - 2
        z = x.to(y.device)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.randint(10, (1, 13), dtype=torch.int)

    a = net(x, y)

    return test_model_formats(net, (x, y), a, "test_Tensor_to")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
