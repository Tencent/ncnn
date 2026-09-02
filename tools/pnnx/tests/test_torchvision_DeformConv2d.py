# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=2*3*3, kernel_size=3)
        self.conv_1 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, kernel_size=3)

        self.conv_2 = nn.Conv2d(in_channels=12, out_channels=3*3, kernel_size=3)
        self.conv_3 = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, kernel_size=3)

    def forward(self, x):
        offset = self.conv_0(x)
        x1 = self.conv_1(x, offset)

        mask = F.sigmoid(self.conv_2(x))
        x2 = self.conv_3(x, offset, mask)
        return x1, x2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)

    a0, a1 = net(x)

    return test_model_formats(
        net,
        (x,),
        (a0, a1),
        "test_torchvision_DeformConv2d",
        unsupported_by_torch_export="torchvision::deform_conv2d",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
