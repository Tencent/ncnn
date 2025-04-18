# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):

        if version.parse(torch.__version__) < version.parse('1.12'):
            x0 = F.interpolate(x, size=60)
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')
            x1 = F.interpolate(x, size=(40), mode='nearest')
            x1 = F.interpolate(x1, scale_factor=(4), mode='nearest')
            x2 = F.interpolate(x, size=60, mode='linear')
            x2 = F.interpolate(x2, scale_factor=2, mode='linear')

            y0 = F.interpolate(y, size=60)
            y0 = F.interpolate(y0, scale_factor=2, mode='nearest')
            y1 = F.interpolate(y, size=(40,40), mode='nearest')
            y1 = F.interpolate(y1, scale_factor=(4,4), mode='nearest')
            y2 = F.interpolate(y, size=(60,50), mode='nearest')
            y2 = F.interpolate(y2, scale_factor=(2,3), mode='nearest')
            y3 = F.interpolate(y, size=60, mode='bilinear')
            y3 = F.interpolate(y3, scale_factor=2, mode='bilinear')

            z0 = F.interpolate(z, size=60)
            z0 = F.interpolate(z0, scale_factor=2, mode='nearest')
            z1 = F.interpolate(z, size=(40,40,40), mode='nearest')
            z1 = F.interpolate(z1, scale_factor=(4,4,4), mode='nearest')
            z2 = F.interpolate(z, size=(60,50,40), mode='nearest')
            z2 = F.interpolate(z2, scale_factor=(2,3,4), mode='nearest')
            z3 = F.interpolate(z, size=60, mode='trilinear')
            z3 = F.interpolate(z3, scale_factor=2, mode='trilinear')

            w = F.interpolate(w, scale_factor=(2.976744,2.976744), mode='nearest', recompute_scale_factor=False)
            return x0, x1, x2, y0, y1, y2, y3, z0, z1, z2, z3, w
        else:
            x = F.interpolate(x, size=16)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = F.interpolate(x, size=(20), mode='nearest')
            x = F.interpolate(x, scale_factor=(4), mode='nearest')
            x = F.interpolate(x, size=16, mode='linear')
            x = F.interpolate(x, scale_factor=2, mode='linear')
            x = F.interpolate(x, size=(24), mode='linear', align_corners=True)
            x = F.interpolate(x, scale_factor=(3), mode='linear', align_corners=True)

            x = F.interpolate(x, scale_factor=1.5, mode='nearest', recompute_scale_factor=True)
            x = F.interpolate(x, scale_factor=1.2, mode='linear', align_corners=False, recompute_scale_factor=True)
            x = F.interpolate(x, scale_factor=0.8, mode='linear', align_corners=True, recompute_scale_factor=True)

            y = F.interpolate(y, size=16)
            y = F.interpolate(y, scale_factor=2, mode='nearest')
            y = F.interpolate(y, size=(20,20), mode='nearest')
            y = F.interpolate(y, scale_factor=(4,4), mode='nearest')
            y = F.interpolate(y, size=(16,24), mode='nearest')
            y = F.interpolate(y, scale_factor=(2,3), mode='nearest')
            y = F.interpolate(y, size=16, mode='bilinear')
            y = F.interpolate(y, scale_factor=2, mode='bilinear')
            y = F.interpolate(y, size=(20,20), mode='bilinear', align_corners=False)
            y = F.interpolate(y, scale_factor=(4,4), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(16,24), mode='bilinear', align_corners=True)
            y = F.interpolate(y, scale_factor=(2,3), mode='bilinear', align_corners=True)

            y = F.interpolate(y, size=16, mode='bicubic')
            y = F.interpolate(y, scale_factor=2, mode='bicubic')
            y = F.interpolate(y, size=(20,20), mode='bicubic', align_corners=False)
            y = F.interpolate(y, scale_factor=(4,4), mode='bicubic', align_corners=False)
            y = F.interpolate(y, size=(16,24), mode='bicubic', align_corners=True)
            y = F.interpolate(y, scale_factor=(2,3), mode='bicubic', align_corners=True)

            y = F.interpolate(y, scale_factor=(1.6,2), mode='nearest', recompute_scale_factor=True)
            y = F.interpolate(y, scale_factor=(2,1.2), mode='bilinear', align_corners=False, recompute_scale_factor=True)
            y = F.interpolate(y, scale_factor=(0.5,0.4), mode='bilinear', align_corners=True, recompute_scale_factor=True)
            y = F.interpolate(y, scale_factor=(0.8,0.9), mode='bicubic', align_corners=False, recompute_scale_factor=True)
            y = F.interpolate(y, scale_factor=(1.1,0.5), mode='bicubic', align_corners=True, recompute_scale_factor=True)

            z = F.interpolate(z, size=16)
            z = F.interpolate(z, scale_factor=2, mode='nearest')
            z = F.interpolate(z, size=(20,20,20), mode='nearest')
            z = F.interpolate(z, scale_factor=(4,4,4), mode='nearest')
            z = F.interpolate(z, size=(16,24,20), mode='nearest')
            z = F.interpolate(z, scale_factor=(2,3,4), mode='nearest')
            z = F.interpolate(z, size=16, mode='trilinear')
            z = F.interpolate(z, scale_factor=2, mode='trilinear')
            z = F.interpolate(z, size=(20,20,20), mode='trilinear', align_corners=False)
            z = F.interpolate(z, scale_factor=(4,4,4), mode='trilinear', align_corners=False)
            z = F.interpolate(z, size=(16,24,20), mode='trilinear', align_corners=True)
            z = F.interpolate(z, scale_factor=(2,3,4), mode='trilinear', align_corners=True)

            z = F.interpolate(z, scale_factor=(1.5,2.5,2), mode='nearest', recompute_scale_factor=True)
            z = F.interpolate(z, scale_factor=(0.7,0.5,1), mode='trilinear', align_corners=False, recompute_scale_factor=True)
            z = F.interpolate(z, scale_factor=(0.9,0.8,1.2), mode='trilinear', align_corners=True, recompute_scale_factor=True)

            w = F.interpolate(w, scale_factor=(2.976744,2.976744), mode='nearest', recompute_scale_factor=False)
            return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32)
    y = torch.rand(1, 3, 32, 32)
    z = torch.rand(1, 3, 32, 32, 32)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, y, z, w)

    # export onnx
    torch.onnx.export(net, (x, y, z, w), "test_F_interpolate.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_interpolate.onnx inputshape=[1,3,32],[1,3,32,32],[1,3,32,32,32],[1,8,86,86]")

    # pnnx inference
    import test_F_interpolate_pnnx
    b = test_F_interpolate_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
