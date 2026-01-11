# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, w):
        if version.parse(torch.__version__) < version.parse('1.12'):
            x = F.upsample_nearest(x, size=(72,128))
            x = F.upsample_nearest(x, scale_factor=2)

            y = F.upsample_nearest(y, size=(20,72,96))
            y = F.upsample_nearest(y, scale_factor=3)
        else:
            x = F.upsample_nearest(x, size=(12,12))
            x = F.upsample_nearest(x, scale_factor=2)

            y = F.upsample_nearest(y, size=(8,10,9))
            y = F.upsample_nearest(y, scale_factor=3)

        w = F.upsample_nearest(w, scale_factor=(2.976744,2.976744))
        return x, y, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    y = torch.rand(1, 4, 10, 24, 32)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, y, w)

    # export onnx
    torch.onnx.export(net, (x, y, w), "test_F_upsample_nearest.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_upsample_nearest.onnx inputshape=[1,12,24,64],[1,4,10,24,32],[1,8,86,86]")

    # pnnx inference
    import test_F_upsample_nearest_pnnx
    b = test_F_upsample_nearest_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
