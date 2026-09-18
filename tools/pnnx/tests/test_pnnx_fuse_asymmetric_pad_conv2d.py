# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pad = nn.ZeroPad2d((1, 2, 1, 2))
        self.conv = nn.Conv2d(4, 4, kernel_size=5, stride=2, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def test():
    net = Model()
    net.eval()

    torch.manual_seed(123)
    with torch.no_grad():
        net.conv.weight.copy_(torch.randn_like(net.conv.weight))
        net.bn.weight.copy_(torch.rand_like(net.bn.weight) * 1.5 + 0.1)
        net.bn.bias.copy_(torch.randn_like(net.bn.bias) * 0.2)
        net.bn.running_mean.copy_(torch.randn_like(net.bn.running_mean) * 0.3)
        net.bn.running_var.copy_(torch.rand_like(net.bn.running_var) * 1.2 + 0.2)

    torch.manual_seed(0)
    x = torch.rand(1, 4, 56, 56)

    a = net(x)
    if a.shape != (1, 4, 28, 28):
        return False

    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_fuse_asymmetric_pad_conv2d.pt")

    os.system("../src/pnnx test_pnnx_fuse_asymmetric_pad_conv2d.pt inputshape=[1,4,56,56]")

    import test_pnnx_fuse_asymmetric_pad_conv2d_pnnx
    b = test_pnnx_fuse_asymmetric_pad_conv2d_pnnx.test_inference()

    if not torch.allclose(a, b, 1e-4, 1e-4):
        return False

    with open("test_pnnx_fuse_asymmetric_pad_conv2d.pnnx.param", "r") as f:
        pnnx_param = f.read()
    if "#2=(1,4,28,28)f32" not in pnnx_param:
        return False

    with open("test_pnnx_fuse_asymmetric_pad_conv2d.ncnn.param", "r") as f:
        ncnn_param = f.read()

    if "ConvolutionDepthWise" in ncnn_param:
        return " 4=1 " in ncnn_param and " 14=1 " in ncnn_param and " 15=2 " in ncnn_param and " 16=2 " in ncnn_param and " 5=1 " in ncnn_param

    return "Padding" in ncnn_param and " 0=1 " in ncnn_param and " 1=2 " in ncnn_param and " 2=1 " in ncnn_param and " 3=2 " in ncnn_param


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
