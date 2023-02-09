# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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
        # norm to -1 ~ 1
        xg1 = xg1 * 2 - 1
        xg2 = xg2 * 2 - 1
        yg1 = yg1 * 2 - 1
        yg2 = yg2 * 2 - 1

        xg1 = torch.permute(xg1, (0, 2, 3, 1))
        xg2 = torch.permute(xg2, (0, 2, 3, 1))
        yg1 = torch.permute(yg1, (0, 2, 3, 4, 1))
        yg2 = torch.permute(yg2, (0, 2, 3, 4, 1))

        x = F.grid_sample(x, xg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        x = F.grid_sample(x, xg2, mode='bilinear', padding_mode='border', align_corners=False)

        y = F.grid_sample(y, yg1, mode='bilinear', padding_mode='zeros', align_corners=False)
        y = F.grid_sample(y, yg2, mode='bilinear', padding_mode='border', align_corners=False)

        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 12, 16)
    xg1 = torch.rand(1, 2, 21, 27)
    xg2 = torch.rand(1, 2, 12, 16)
    y = torch.rand(1, 5, 10, 12, 16)
    yg1 = torch.rand(1, 3, 10, 21, 27)
    yg2 = torch.rand(1, 3, 10, 12, 16)

    a0, a1 = net(x, xg1, xg2, y, yg1, yg2)

    # export torchscript
    mod = torch.jit.trace(net, (x, xg1, xg2, y, yg1, yg2))
    mod.save("test_pnnx_fuse_permute_gridsample.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_permute_gridsample.pt inputshape=[1,3,12,16],[1,2,21,27],[1,2,12,16],[1,5,10,12,16],[1,3,10,21,27],[1,3,10,12,16]")

    # pnnx inference
    import test_pnnx_fuse_permute_gridsample_pnnx
    b0, b1 = test_pnnx_fuse_permute_gridsample_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
