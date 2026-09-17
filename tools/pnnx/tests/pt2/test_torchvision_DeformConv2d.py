# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=2*3*3, kernel_size=3)
        self.conv_1 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, kernel_size=3)

        self.conv_2 = nn.Conv2d(in_channels=12, out_channels=3*3, kernel_size=3)
        self.conv_3 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, kernel_size=3)
        self.conv_4 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, kernel_size=3, bias=False)
        self.conv_5 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=15, kernel_size=3, bias=False)

    def forward(self, x):
        offset = self.conv_0(x)
        x1 = self.conv_1(x, offset)

        mask = F.sigmoid(self.conv_2(x))
        x2 = self.conv_3(x, offset, mask)
        x3 = self.conv_4(x, offset)
        x4 = self.conv_5(x, offset, mask)
        return x1, x2, x3, x4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)

    a = net(x)

    # export pt2
    mod = torch.export.export(net, (x,))
    torch.export.save(mod, "test_torchvision_DeformConv2d.pt2")

    # pt2 to pnnx
    import os
    os.system(os.path.normpath("../../src/pnnx") + " test_torchvision_DeformConv2d.pt2 inputshape=[1,12,64,64]")

    # pnnx inference
    import test_torchvision_DeformConv2d_pnnx
    b = test_torchvision_DeformConv2d_pnnx.test_inference()

    return all(torch.equal(x, y) for x, y in zip(a, b))

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
