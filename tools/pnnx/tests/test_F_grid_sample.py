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

    def forward(self, x, xg1, xg2, y, yg1, yg2):
        x = F.grid_sample(x, xg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        x = F.grid_sample(x, xg2, mode='bilinear', padding_mode='border', align_corners=False)
        x = F.grid_sample(x, xg1, mode='bilinear', padding_mode='reflection', align_corners=False)
        x = F.grid_sample(x, xg2, mode='nearest', padding_mode='zeros', align_corners=False)
        x = F.grid_sample(x, xg1, mode='nearest', padding_mode='border', align_corners=False)
        x = F.grid_sample(x, xg2, mode='nearest', padding_mode='reflection', align_corners=False)
        x = F.grid_sample(x, xg1, mode='bicubic', padding_mode='zeros', align_corners=False)
        x = F.grid_sample(x, xg2, mode='bicubic', padding_mode='border', align_corners=False)
        x = F.grid_sample(x, xg1, mode='bicubic', padding_mode='reflection', align_corners=False)
        x = F.grid_sample(x, xg2, mode='bilinear', padding_mode='zeros', align_corners=True)
        x = F.grid_sample(x, xg1, mode='bilinear', padding_mode='border', align_corners=True)
        x = F.grid_sample(x, xg2, mode='bilinear', padding_mode='reflection', align_corners=True)
        x = F.grid_sample(x, xg1, mode='nearest', padding_mode='zeros', align_corners=True)
        x = F.grid_sample(x, xg2, mode='nearest', padding_mode='border', align_corners=True)
        x = F.grid_sample(x, xg1, mode='nearest', padding_mode='reflection', align_corners=True)
        x = F.grid_sample(x, xg2, mode='bicubic', padding_mode='zeros', align_corners=True)
        x = F.grid_sample(x, xg1, mode='bicubic', padding_mode='border', align_corners=True)
        x = F.grid_sample(x, xg2, mode='bicubic', padding_mode='reflection', align_corners=True)

        y = F.grid_sample(y, yg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        y = F.grid_sample(y, yg2, mode='bilinear', padding_mode='border', align_corners=False)
        y = F.grid_sample(y, yg1, mode='bilinear', padding_mode='reflection', align_corners=False)
        y = F.grid_sample(y, yg2, mode='nearest', padding_mode='zeros', align_corners=False)
        y = F.grid_sample(y, yg1, mode='nearest', padding_mode='border', align_corners=False)
        y = F.grid_sample(y, yg2, mode='nearest', padding_mode='reflection', align_corners=False)
        y = F.grid_sample(y, yg1, mode='bilinear', padding_mode='zeros', align_corners=True)
        y = F.grid_sample(y, yg2, mode='bilinear', padding_mode='border', align_corners=True)
        y = F.grid_sample(y, yg1, mode='bilinear', padding_mode='reflection', align_corners=True)
        y = F.grid_sample(y, yg2, mode='nearest', padding_mode='zeros', align_corners=True)
        y = F.grid_sample(y, yg1, mode='nearest', padding_mode='border', align_corners=True)
        y = F.grid_sample(y, yg2, mode='nearest', padding_mode='reflection', align_corners=True)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 12, 16)
    xg1 = torch.rand(1, 21, 27, 2)
    xg2 = torch.rand(1, 12, 16, 2)
    y = torch.rand(1, 5, 10, 12, 16)
    yg1 = torch.rand(1, 10, 21, 27, 3)
    yg2 = torch.rand(1, 10, 12, 16, 3)

    a0, a1 = net(x, xg1, xg2, y, yg1, yg2)

    # export torchscript
    mod = torch.jit.trace(net, (x, xg1, xg2, y, yg1, yg2))
    mod.save("test_F_grid_sample.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_grid_sample.pt inputshape=[1,3,12,16],[1,21,27,2],[1,12,16,2],[1,5,10,12,16],[1,10,21,27,3],[1,10,12,16,3]")

    # pnnx inference
    import test_F_grid_sample_pnnx
    b0, b1 = test_F_grid_sample_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
