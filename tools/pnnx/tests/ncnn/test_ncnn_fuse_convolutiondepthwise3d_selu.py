# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1, groups=12)
        self.act = nn.SELU()

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 16, 52, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_ncnn_fuse_convolutiondepthwise3d_selu.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_convolutiondepthwise3d_selu.pt inputshape=[1,12,16,52,64]")

    # ncnn inference
    import test_ncnn_fuse_convolutiondepthwise3d_selu_ncnn
    b = test_ncnn_fuse_convolutiondepthwise3d_selu_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
