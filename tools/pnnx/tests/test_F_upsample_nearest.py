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
        x = F.upsample_nearest(x, size=(12,12))
        x = F.upsample_nearest(x, scale_factor=2)

        y = F.upsample_nearest(y, size=(8,10,9))
        y = F.upsample_nearest(y, scale_factor=3)

        w = F.upsample_nearest(w, scale_factor=(2.976744,2.976744))
        return x, y, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    y = torch.rand(1, 4, 10, 24, 32)
    w = torch.rand(1, 8, 86, 86)

    a = net(x, y, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, w))
    mod.save("test_F_upsample_nearest.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_upsample_nearest.pt inputshape=[1,12,24,64],[1,4,10,24,32],[1,8,86,86]")

    # pnnx inference
    import test_F_upsample_nearest_pnnx
    b = test_F_upsample_nearest_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
