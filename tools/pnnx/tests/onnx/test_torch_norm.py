# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.norm(x)
        y = torch.norm(y, p=2, dim=(2,3), keepdim=False)
        z = torch.norm(z, p=1, dim=(-2,), keepdim=True)
        return x, y, z

def test():
    # old torch throws error
    # RuntimeError: Attribute 'axes' is expected to have field 'ints'
    # ==> Context: Bad node spec for node. Name: ReduceSum_1 OpType: ReduceSum
    if version.parse(torch.__version__) < version.parse('1.12'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_torch_norm.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_torch_norm.onnx inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]")

    # pnnx inference
    import test_torch_norm_pnnx
    b = test_torch_norm_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
