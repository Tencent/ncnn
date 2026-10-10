# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x, output_size=8)
        x = F.adaptive_avg_pool1d(x, output_size=1)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_F_adaptive_avg_pool1d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_avg_pool1d.onnx inputshape=[1,12,24]")

    # pnnx inference
    import test_F_adaptive_avg_pool1d_pnnx
    b = test_F_adaptive_avg_pool1d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
