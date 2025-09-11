# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.squeeze(x, 1)
        y = torch.squeeze(y)
        z = torch.squeeze(z, 4)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, 16)
    y = torch.rand(1, 5, 1, 11)
    z = torch.rand(14, 8, 5, 9, 1)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_torch_squeeze.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_torch_squeeze.onnx inputshape=[1,1,16],[1,5,1,11],[14,8,5,9,1]")

    # pnnx inference
    import test_torch_squeeze_pnnx
    b = test_torch_squeeze_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
