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
        out0 = F.adaptive_avg_pool3d(x, output_size=(8,11,16))
        out1 = F.adaptive_avg_pool3d(x, output_size=1)
        if version.parse(torch.__version__) < version.parse('1.10'):
            return out0, out1

        out2 = F.adaptive_avg_pool3d(x, output_size=(None,3,4))
        out3 = F.adaptive_avg_pool3d(x, output_size=(6,None,None))
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 33, 64)

    a = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_F_adaptive_avg_pool3d.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_F_adaptive_avg_pool3d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_avg_pool3d.onnx inputshape=[1,12,24,33,64]")

    # pnnx inference
    import test_F_adaptive_avg_pool3d_pnnx
    b = test_F_adaptive_avg_pool3d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
