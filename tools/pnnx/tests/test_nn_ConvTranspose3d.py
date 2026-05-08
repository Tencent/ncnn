# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.deconv_0 = nn.ConvTranspose3d(in_channels=12, out_channels=16, kernel_size=3)
        self.deconv_1 = nn.ConvTranspose3d(in_channels=16, out_channels=20, kernel_size=(2,3,4), stride=(2,2,1), padding=2, output_padding=0)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=20, out_channels=24, kernel_size=(1,2,3), stride=1, padding=(2,3,4), output_padding=(0,0,0), dilation=1, groups=1, bias=False)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=24, out_channels=28, kernel_size=(5,4,3), stride=2, padding=0, output_padding=(0,1,1), dilation=1, groups=4, bias=True)
        self.deconv_4 = nn.ConvTranspose3d(in_channels=28, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=(1,2,2), groups=2, bias=False)
        self.deconv_5 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=3, output_padding=1, dilation=1, groups=32, bias=True)
        self.deconv_6 = nn.ConvTranspose3d(in_channels=32, out_channels=28, kernel_size=2, stride=1, padding=2, output_padding=0, dilation=1, groups=1, bias=False)
        self.deconv_7 = nn.ConvTranspose3d(in_channels=28, out_channels=24, kernel_size=3, stride=2, padding=(5,6,7), output_padding=(1,0,1), dilation=2, groups=1, bias=True)

        if version.parse(torch.__version__) < version.parse('2.1'):
            self.deconv_7 = torch.nn.utils.weight_norm(self.deconv_7)
        else:
            self.deconv_7 = torch.nn.utils.parametrizations.weight_norm(self.deconv_7)

        self.downsample = nn.Conv3d(24, 16, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(16, 24, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.deconv_0(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.deconv_6(x)
        x = self.deconv_7(x)

        y = self.downsample(x)
        x = self.upsample(y, output_size=x.size())

        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 7, 7, 10)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_ConvTranspose3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_ConvTranspose3d.pt inputshape=[1,12,7,7,10]")

    # pnnx inference
    import test_nn_ConvTranspose3d_pnnx
    b = test_nn_ConvTranspose3d_pnnx.test_inference()

    return torch.allclose(a, b, 1e-3, 1e-3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
