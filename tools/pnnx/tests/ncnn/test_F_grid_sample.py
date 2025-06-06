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

    def forward(self, x, xg1, xg2, xgp1, xgp2, y, yg1, yg2, ygp1, ygp2):
        # norm to -1 ~ 1
        xg1 = xg1 * 2 - 1
        xg2 = xg2 * 2 - 1
        yg1 = yg1 * 2 - 1
        yg2 = yg2 * 2 - 1

        x0 = F.grid_sample(x, xg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        x0 = F.grid_sample(x0, xg2, mode='bilinear', padding_mode='border', align_corners=False)
        x0 = F.grid_sample(x0, xg1, mode='bilinear', padding_mode='reflection', align_corners=False)
        x0 = F.grid_sample(x0, xg2, mode='nearest', padding_mode='zeros', align_corners=False)
        x0 = F.grid_sample(x0, xg1, mode='nearest', padding_mode='border', align_corners=False)
        x0 = F.grid_sample(x0, xg2, mode='nearest', padding_mode='reflection', align_corners=False)
        x0 = F.grid_sample(x0, xg1, mode='bicubic', padding_mode='zeros', align_corners=False)
        x0 = F.grid_sample(x0, xg2, mode='bicubic', padding_mode='border', align_corners=False)
        x0 = F.grid_sample(x0, xg1, mode='bicubic', padding_mode='reflection', align_corners=False)
        x0 = F.grid_sample(x0, xg2, mode='bilinear', padding_mode='zeros', align_corners=True)
        x0 = F.grid_sample(x0, xg1, mode='bilinear', padding_mode='border', align_corners=True)
        x0 = F.grid_sample(x0, xg2, mode='bilinear', padding_mode='reflection', align_corners=True)
        x0 = F.grid_sample(x0, xg1, mode='nearest', padding_mode='zeros', align_corners=True)
        x0 = F.grid_sample(x0, xg2, mode='nearest', padding_mode='border', align_corners=True)
        x0 = F.grid_sample(x0, xg1, mode='nearest', padding_mode='reflection', align_corners=True)
        x0 = F.grid_sample(x0, xg2, mode='bicubic', padding_mode='zeros', align_corners=True)
        x0 = F.grid_sample(x0, xg1, mode='bicubic', padding_mode='border', align_corners=True)
        x0 = F.grid_sample(x0, xg2, mode='bicubic', padding_mode='reflection', align_corners=True)

        y0 = F.grid_sample(y, yg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        y0 = F.grid_sample(y0, yg2, mode='bilinear', padding_mode='border', align_corners=False)
        y0 = F.grid_sample(y0, yg1, mode='bilinear', padding_mode='reflection', align_corners=False)
        y0 = F.grid_sample(y0, yg2, mode='nearest', padding_mode='zeros', align_corners=False)
        y0 = F.grid_sample(y0, yg1, mode='nearest', padding_mode='border', align_corners=False)
        y0 = F.grid_sample(y0, yg2, mode='nearest', padding_mode='reflection', align_corners=False)
        y0 = F.grid_sample(y0, yg1, mode='bilinear', padding_mode='zeros', align_corners=True)
        y0 = F.grid_sample(y0, yg2, mode='bilinear', padding_mode='border', align_corners=True)
        y0 = F.grid_sample(y0, yg1, mode='bilinear', padding_mode='reflection', align_corners=True)
        y0 = F.grid_sample(y0, yg2, mode='nearest', padding_mode='zeros', align_corners=True)
        y0 = F.grid_sample(y0, yg1, mode='nearest', padding_mode='border', align_corners=True)
        y0 = F.grid_sample(y0, yg2, mode='nearest', padding_mode='reflection', align_corners=True)

        xgp1 = xgp1.permute(0, 2, 3, 1)
        xgp2 = xgp2.permute(0, 2, 3, 1)
        ygp1 = ygp1.permute(0, 2, 3, 4, 1)
        ygp2 = ygp2.permute(0, 2, 3, 4, 1)

        x1 = F.grid_sample(x, xgp1, mode='bilinear', padding_mode='zeros', align_corners=False)
        x1 = F.grid_sample(x1, xgp2, mode='bilinear', padding_mode='border', align_corners=False)

        y1 = F.grid_sample(y, ygp1, mode='bilinear', padding_mode='zeros', align_corners=False)
        y1 = F.grid_sample(y1, ygp2, mode='bilinear', padding_mode='border', align_corners=False)
        return x0, y0, x1, y1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 12, 16)
    xg1 = torch.rand(1, 21, 27, 2)
    xg2 = torch.rand(1, 12, 16, 2)
    xgp1 = torch.rand(1, 2, 21, 27)
    xgp2 = torch.rand(1, 2, 12, 16)
    y = torch.rand(1, 5, 10, 12, 16)
    yg1 = torch.rand(1, 10, 21, 27, 3)
    yg2 = torch.rand(1, 10, 12, 16, 3)
    ygp1 = torch.rand(1, 3, 10, 21, 27)
    ygp2 = torch.rand(1, 3, 10, 12, 16)

    a0, a1, a2, a3 = net(x, xg1, xg2, xgp1, xgp2, y, yg1, yg2, ygp1, ygp2)

    # export torchscript
    mod = torch.jit.trace(net, (x, xg1, xg2, xgp1, xgp2, y, yg1, yg2, ygp1, ygp2))
    mod.save("test_F_grid_sample.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_grid_sample.pt inputshape=[1,3,12,16],[1,21,27,2],[1,12,16,2],[1,2,21,27],[1,2,12,16],[1,5,10,12,16],[1,10,21,27,3],[1,10,12,16,3],[1,3,10,21,27],[1,3,10,12,16]")

    # ncnn inference
    import test_F_grid_sample_ncnn
    b0, b1, b2, b3 = test_F_grid_sample_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-6, 1e-6) and torch.allclose(a1, b1, 1e-6, 1e-6) and torch.allclose(a2, b2, 1e-6, 1e-6) and torch.allclose(a3, b3, 1e-6, 1e-6)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
