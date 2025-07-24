# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pad_0 = nn.ConstantPad1d(2, 0.1)
        self.pad_1 = nn.ConstantPad1d(padding=(3,4), value=-1)
        self.pad_2 = nn.ConstantPad1d(padding=(1,0), value=123)

    def forward(self, x):
        x = self.pad_0(x)
        x = self.pad_1(x)
        x = self.pad_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x, ), "test_nn_ConstantPad1d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_ConstantPad1d.onnx inputshape=[1,12,13]")

    # pnnx inference
    import test_nn_ConstantPad1d_pnnx
    b = test_nn_ConstantPad1d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
