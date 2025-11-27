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
        x, indices = F.adaptive_max_pool1d(x, output_size=6, return_indices=True)
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x, indices

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a0, a1 = net(x)

    # export onnx
    if version.parse(torch.__version__) >= version.parse('2.9') and version.parse(torch.__version__) < version.parse('2.10'):
        torch.onnx.export(net, (x,), "test_F_adaptive_max_pool1d.onnx", dynamo=False)
    else:
        torch.onnx.export(net, (x,), "test_F_adaptive_max_pool1d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_max_pool1d.onnx inputshape=[1,12,24]")

    # pnnx inference
    import test_F_adaptive_max_pool1d_pnnx
    b0, b1 = test_F_adaptive_max_pool1d_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
