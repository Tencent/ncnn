# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import torch.nn as nn
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x2d = torch.rand(1, 3, 8, 8)
    x1d = torch.rand(1, 3, 16)

    a = net(x2d), net(x1d)

    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x2d,), "test_nn_Resize_nearest.onnx", dynamo=False, input_names=["x2d"])
    else:
        torch.onnx.export(net, (x2d,), "test_nn_Resize_nearest.onnx", dynamo=False, input_names=["x2d"])

    import os
    os.system("../../src/pnnx test_nn_Resize_nearest.onnx inputshape=[1,3,8,8] fp16=0")

    import test_nn_Resize_nearest_pnnx
    b = test_nn_Resize_nearest_pnnx.test_inference()

    import test_nn_Resize_nearest_ncnn
    c = test_nn_Resize_nearest_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
