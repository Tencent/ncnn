# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1, groups=2)
        self.act_0 = nn.ELU()
        self.act_1 = nn.ELU(alpha=0.5)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act_0(x)
        x = self.conv_1(x)
        x = self.act_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 52, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_ncnn_fuse_convolution_elu.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_convolution_elu.pt inputshape=[1,12,52,64]")

    # ncnn inference
    import test_ncnn_fuse_convolution_elu_ncnn
    b = test_ncnn_fuse_convolution_elu_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
