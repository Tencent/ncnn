# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w3 = nn.Parameter(torch.rand(16))
        self.b3 = nn.Parameter(torch.rand(16))
        self.w4 = nn.Parameter(torch.rand(12))
        self.b4 = nn.Parameter(torch.rand(12))
        self.w5 = nn.Parameter(torch.rand(32))
        self.b5 = nn.Parameter(torch.rand(32))

    def forward(self, x, y, z, w0, b0, w1, b1, w2, b2):
        x = F.group_norm(x, 2, w0, b0)
        x = F.group_norm(x, 1, None, None)
        x = F.group_norm(x, 4, self.w3, self.b3)

        y = F.group_norm(y, 3, w1, b1, eps=1e-4)
        y = F.group_norm(y, 4, None, None)
        y = F.group_norm(y, 6, self.w4, self.b4)

        z = F.group_norm(z, 32, w2, b2)
        z = F.group_norm(z, 4, None, None, eps=1e-2)
        z = F.group_norm(z, 8, self.w5, self.b5)
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('2.6'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 12, 16)
    z = torch.rand(1, 32, 12, 16)
    w0 = torch.rand(16)
    b0 = torch.rand(16)
    w1 = torch.rand(12)
    b1 = torch.rand(12)
    w2 = torch.rand(32)
    b2 = torch.rand(32)

    a = net(x, y, z, w0, b0, w1, b1, w2, b2)

    # export onnx
    torch.onnx.export(net, (x, y, z, w0, b0, w1, b1, w2, b2), "test_F_group_norm.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_group_norm.onnx inputshape=[1,16],[12,12,16],[1,32,12,16],[16],[16],[12],[12],[32],[32]")

    # pnnx inference
    import test_F_group_norm_pnnx
    b = test_F_group_norm_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
