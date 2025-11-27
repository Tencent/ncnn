# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, groups=12)
        self.conv_2 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=3, padding=2, groups=2)

    def forward(self, x):
        x = F.pad(x, (2,0), mode='constant', value=0)
        x = self.conv_0(x)
        x = F.pad(x, (3,4), mode='constant', value=2.3)
        x = self.conv_1(x)
        x = F.pad(x, (0,1), mode='constant', value=1)
        x = self.conv_2(x)
        return x

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 30, 30)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, (x, ))
    mod.save("test_ncnn_fuse_pad_conv.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_pad_conv.pt inputshape=[1,3,30,30]")

    # ncnn inference
    import test_ncnn_fuse_pad_conv_ncnn
    b = test_ncnn_fuse_pad_conv_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
