# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

        self.up_1d_0_0 = nn.Upsample(size=16)
        self.up_1d_0_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_1d_0_2 = nn.Upsample(size=(20), mode='nearest')
        self.up_1d_0_3 = nn.Upsample(scale_factor=(4), mode='nearest')
        self.up_1d_1_0 = nn.Upsample(size=16, mode='linear')
        self.up_1d_1_1 = nn.Upsample(scale_factor=2, mode='linear')
        self.up_1d_1_2 = nn.Upsample(size=(24), mode='linear', align_corners=True)
        self.up_1d_1_3 = nn.Upsample(scale_factor=(3), mode='linear', align_corners=True)

        self.up_2d_0_0 = nn.Upsample(size=16)
        self.up_2d_0_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_2d_0_2 = nn.Upsample(size=(20,20), mode='nearest')
        self.up_2d_0_3 = nn.Upsample(scale_factor=(4,4), mode='nearest')
        self.up_2d_0_4 = nn.Upsample(size=(16,24), mode='nearest')
        self.up_2d_0_5 = nn.Upsample(scale_factor=(2,3), mode='nearest')
        self.up_2d_1_0 = nn.Upsample(size=16, mode='bilinear')
        self.up_2d_1_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2d_1_2 = nn.Upsample(size=(20,20), mode='bilinear', align_corners=False)
        self.up_2d_1_3 = nn.Upsample(scale_factor=(4,4), mode='bilinear', align_corners=False)
        self.up_2d_1_4 = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.up_2d_1_5 = nn.Upsample(scale_factor=(2,3), mode='bilinear', align_corners=True)
        self.up_2d_2_0 = nn.Upsample(size=16, mode='bicubic')
        self.up_2d_2_1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_2d_2_2 = nn.Upsample(size=(20,20), mode='bicubic', align_corners=False)
        self.up_2d_2_3 = nn.Upsample(scale_factor=(4,4), mode='bicubic', align_corners=False)
        self.up_2d_2_4 = nn.Upsample(size=(16,24), mode='bicubic', align_corners=True)
        self.up_2d_2_5 = nn.Upsample(scale_factor=(2,3), mode='bicubic', align_corners=True)

        self.up_w = nn.Upsample(scale_factor=(1.499,1.499), mode='nearest')

    def forward(self, x, y, w):
        x = self.up_1d_0_0(x)
        x = self.up_1d_0_1(x)
        x = self.up_1d_0_2(x)
        x = self.up_1d_0_3(x)
        x = self.up_1d_1_0(x)
        x = self.up_1d_1_1(x)
        x = self.up_1d_1_2(x)
        x = self.up_1d_1_3(x)

        y = self.up_2d_0_0(y)
        y = self.up_2d_0_1(y)
        y = self.up_2d_0_2(y)
        y = self.up_2d_0_3(y)
        y = self.up_2d_0_4(y)
        y = self.up_2d_0_5(y)
        y = self.up_2d_1_0(y)
        y = self.up_2d_1_1(y)
        y = self.up_2d_1_2(y)
        y = self.up_2d_1_3(y)
        y = self.up_2d_1_4(y)
        y = self.up_2d_1_5(y)
        y = self.up_2d_2_0(y)
        y = self.up_2d_2_1(y)
        y = self.up_2d_2_2(y)
        y = self.up_2d_2_3(y)
        y = self.up_2d_2_4(y)
        y = self.up_2d_2_5(y)

        w = self.up_w(w)

        return x, y, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32)
    y = torch.rand(1, 3, 32, 32)
    w = torch.rand(1, 8, 12, 12)

    a = net(x, y, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, w))
    mod.save("test_nn_Upsample.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Upsample.pt inputshape=[1,3,32],[1,3,32,32],[1,8,12,12]")

    # ncnn inference
    import test_nn_Upsample_ncnn
    b = test_nn_Upsample_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
