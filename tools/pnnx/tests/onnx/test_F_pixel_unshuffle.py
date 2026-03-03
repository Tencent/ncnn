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
        x = F.pixel_unshuffle(x, 4)
        x = F.pixel_unshuffle(x, 2)
        return x

def test():
    if version.parse(torch.__version__) < version.parse('1.12'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)

    a = net(x)

    # export onnx
    torch.onnx.export(net, x, "test_F_pixel_unshuffle.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_pixel_unshuffle.onnx inputshape=[1,3,128,128]")

    # pnnx inference
    import test_F_pixel_unshuffle_pnnx
    b = test_F_pixel_unshuffle_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
