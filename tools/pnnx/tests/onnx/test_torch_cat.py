# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.cat((x, y), dim=1)
        out1 = torch.cat((z, w), dim=3)
        out2 = torch.cat((w, w), dim=2)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 2, 16)
    z = torch.rand(1, 5, 9, 11)
    w = torch.rand(1, 5, 9, 3)

    a = net(x, y, z, w)

    # export onnx
    torch.onnx.export(net, (x, y, z, w), "test_torch_cat.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_torch_cat.onnx inputshape=[1,3,16],[1,2,16],[1,5,9,11],[1,5,9,3]")

    # pnnx inference
    import test_torch_cat_pnnx
    b = test_torch_cat_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
