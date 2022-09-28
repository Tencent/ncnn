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

    def forward(self, x, y, w):
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

        w = F.interpolate(w, scale_factor=(2.976744,2.976744), mode='nearest', recompute_scale_factor=False)
        return x, y, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 32)
    y = torch.rand(1, 3, 32, 32)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, y, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, w))
    mod.save("test_F_interpolate.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_interpolate.pt inputshape=[1,3,32],[1,3,32,32],[1,8,86,86]")

    # ncnn inference
    import test_F_interpolate_ncnn
    b = test_F_interpolate_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
