# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.deconv_0 = nn.ConvTranspose3d(in_channels=12, out_channels=16, kernel_size=3)
        self.bn_0 = nn.BatchNorm3d(num_features=16)
        self.deconv_1 = nn.ConvTranspose3d(in_channels=16, out_channels=20, kernel_size=(2,4,2), stride=(2,1,2), padding=2, output_padding=0)
        self.bn_1 = nn.BatchNorm3d(num_features=20)
        self.deconv_2 = nn.ConvTranspose3d(in_channels=20, out_channels=24, kernel_size=(1,3,3), stride=1, padding=(2,4,4), output_padding=(0,0,0), dilation=1, groups=1, bias=False)
        self.bn_2 = nn.BatchNorm3d(num_features=24, eps=1e-1, affine=False)
        self.deconv_3 = nn.ConvTranspose3d(in_channels=24, out_channels=28, kernel_size=(5,4,5), stride=2, padding=0, output_padding=(0,1,0), dilation=1, groups=4, bias=True)
        self.bn_3 = nn.BatchNorm3d(num_features=28, eps=1e-1, affine=False)
        self.deconv_4 = nn.ConvTranspose3d(in_channels=28, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=(1,2,2), groups=2, bias=False)
        self.bn_4 = nn.BatchNorm3d(num_features=32)
        self.deconv_5 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=3, output_padding=1, dilation=1, groups=32, bias=True)
        self.bn_5 = nn.BatchNorm3d(num_features=32)
        self.deconv_6 = nn.ConvTranspose3d(in_channels=32, out_channels=28, kernel_size=2, stride=1, padding=2, output_padding=0, dilation=1, groups=1, bias=False)
        self.bn_6 = nn.BatchNorm3d(num_features=28, affine=True)
        self.deconv_7 = nn.ConvTranspose3d(in_channels=28, out_channels=24, kernel_size=3, stride=2, padding=(5,6,6), output_padding=(1,0,0), dilation=2, groups=1, bias=True)
        self.bn_7 = nn.BatchNorm3d(num_features=24, affine=True)

    def forward(self, x):
        x = self.deconv_0(x)
        x = self.bn_0(x)
        x = self.deconv_1(x)
        x = self.bn_1(x)
        x = self.deconv_2(x)
        x = self.bn_2(x)
        x = self.deconv_3(x)
        x = self.bn_3(x)
        x = self.deconv_4(x)
        x = self.bn_4(x)
        x = self.deconv_5(x)
        x = self.bn_5(x)
        x = self.deconv_6(x)
        x = self.bn_6(x)
        x = self.deconv_7(x)
        x = self.bn_7(x)

        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 10, 10, 10)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_fuse_convtranspose3d_batchnorm3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_convtranspose3d_batchnorm3d.pt inputshape=[1,12,10,10,10]")

    # pnnx inference
    import test_pnnx_fuse_convtranspose3d_batchnorm3d_pnnx
    b = test_pnnx_fuse_convtranspose3d_batchnorm3d_pnnx.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
