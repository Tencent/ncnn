# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.clone(x, memory_format=torch.contiguous_format)
        y = torch.clone(y, memory_format=torch.channels_last)
        z = torch.clone(z, memory_format=torch.preserve_format)
        default = torch.clone(x.transpose(1, 2))
        channels_last_3d = torch.clone(z, memory_format=torch.channels_last_3d)
        return x, y, z, default, channels_last_3d

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    mod = convert_and_import(
        net,
        (x, y, z),
        "test_torch_clone",
        pnnx_args=("inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]",),
    )

    b = mod.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape or a0.dtype != b0.dtype or a0.stride() != b0.stride() or not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
