# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pad_0 = nn.ConstantPad1d(2, 0.0)
        self.pad_1 = nn.ReflectionPad1d(4)
        self.pad_2 = nn.ReplicationPad1d(3)

        self.conv_0 = nn.Conv1d(in_channels=12, out_channels=14, kernel_size=3)
        self.conv_1 = nn.Conv1d(in_channels=14, out_channels=14, kernel_size=1)
        self.conv_2 = nn.Conv1d(in_channels=14, out_channels=14, kernel_size=2)
        self.conv_3 = nn.Conv1d(in_channels=14, out_channels=12, kernel_size=3, padding=(1,))

    def forward(self, x):
        x = self.pad_0(x)
        x = F.pad(x, pad=(1,1))
        x = self.conv_0(x)

        x = self.pad_1(x)
        x = self.conv_1(x)

        x = F.pad(x, pad=(3,3), mode='reflect')
        x = self.conv_1(x)

        x = self.pad_2(x)
        x = self.conv_2(x)

        x = F.pad(x, pad=(1,1), mode='replicate')
        x = self.conv_2(x)

        x = F.pad(x, pad=(2,2,0,0,0,0))
        x = self.conv_3(x)

        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_pnnx_fuse_pad_conv1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_pnnx_fuse_pad_conv1d.pt inputshape=[1,12,13]")

    # pnnx inference
    import test_pnnx_pnnx_fuse_pad_conv1d_pnnx
    b = test_pnnx_pnnx_fuse_pad_conv1d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
