# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=3)
        x = F.avg_pool2d(x, kernel_size=4, stride=2, padding=2)
        x = F.avg_pool2d(x, kernel_size=(1,3), stride=1, padding=(0,1), ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=(4,5), stride=(1,2), padding=(1,2), ceil_mode=True, count_include_pad=False)
        x = F.avg_pool2d(x, kernel_size=(5,3), stride=(2,1), padding=1, ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        # x = F.avg_pool2d(x, kernel_size=(5,4), stride=1, padding=2, ceil_mode=False, count_include_pad=False, divisor_override=18)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128, 127)

    a = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_F_avg_pool2d.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_avg_pool2d.onnx inputshape=[1,12,128,127]")

    # pnnx inference
    import test_F_avg_pool2d_pnnx
    b = test_F_avg_pool2d_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
