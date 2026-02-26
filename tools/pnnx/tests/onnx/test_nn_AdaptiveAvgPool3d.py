# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveAvgPool3d(output_size=(9,4,5))
        self.pool_1 = nn.AdaptiveAvgPool3d(output_size=1)
        self.pool_2 = nn.AdaptiveAvgPool3d(output_size=(None,4,3))
        self.pool_3 = nn.AdaptiveAvgPool3d(output_size=(2,None,None))

    def forward(self, x):
        out0 = self.pool_0(x)
        out1 = self.pool_1(x)
        if version.parse(torch.__version__) < version.parse('1.10'):
            return out0, out1

        out2 = self.pool_2(x)
        out3 = self.pool_3(x)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 18, 16, 15)

    a = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_nn_AdaptiveAvgPool3d.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_nn_AdaptiveAvgPool3d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_AdaptiveAvgPool3d.onnx inputshape=[1,128,18,16,15]")

    # pnnx inference
    import test_nn_AdaptiveAvgPool3d_pnnx
    b = test_nn_AdaptiveAvgPool3d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
