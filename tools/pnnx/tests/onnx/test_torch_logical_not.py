# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        a = torch.ge(x, y)
        out = torch.logical_not(a)
        return out

def test():
    if version.parse(torch.__version__) < version.parse('2.1'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_torch_logical_not.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_torch_logical_not.onnx inputshape=[3,16],[3,16]")

    # pnnx inference
    import test_torch_logical_not_pnnx
    b = test_torch_logical_not_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
