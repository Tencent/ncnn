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

    def forward(self, x, y):
        z1 = F.pairwise_distance(x,y,p=1,keepdim=False)
        z2 = F.pairwise_distance(x,y,p=2,keepdim=True)
        z3 = F.pairwise_distance(x,y)
        z4 = F.pairwise_distance(x,y,eps = 1e-3)
        return z1,z2,z3,z4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 128, 128)
    y = torch.rand(12, 128, 128)

    a0,a1,a2,a3 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_pairwise_distance.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_pairwise_distance.pt inputshape=[12,128,128],[12,128,128]")

    # pnnx inference
    import test_F_pairwise_distance_pnnx
    b0,b1,b2,b3 = test_F_pairwise_distance_pnnx.test_inference()

    return torch.equal(a0,b0) and torch.equal(a1,b1) and torch.equal(a2,b2) and torch.equal(a3,b3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
