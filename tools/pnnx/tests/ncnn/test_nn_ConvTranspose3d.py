# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        x = self.deconv_0(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.deconv_6(x)
        x = self.deconv_7(x)

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
    os.system("../../src/pnnx test_nn_ConvTranspose3d.pt inputshape=[1,12,7,7,10]")

    # ncnn inference
    import test_nn_ConvTranspose3d_ncnn
    b = test_nn_ConvTranspose3d_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
