# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0, indices0 = F.adaptive_max_pool2d(x, output_size=(8,8), return_indices=True)
        out1 = F.adaptive_max_pool2d(x, output_size=1)
        if version.parse(torch.__version__) < version.parse('1.10'):
            return out0, indices0, out1

        out2 = F.adaptive_max_pool2d(x, output_size=(None,4))
        out3, indices3 = F.adaptive_max_pool2d(x, output_size=(6,None), return_indices=True)
        return out0, indices0, out1, out2, out3, indices3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_F_adaptive_max_pool2d.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_F_adaptive_max_pool2d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_max_pool2d.onnx inputshape=[1,12,24,64]")

    # pnnx inference
    import test_F_adaptive_max_pool2d_pnnx
    b = test_F_adaptive_max_pool2d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
