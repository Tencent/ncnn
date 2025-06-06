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

    def forward(self, x, y, z, w):
        x = F.pad(x, (3,4), mode='constant', value=1.3)
        x = F.pad(x, (2,2))

        y = F.pad(y, (5,6), mode='reflect')
        y = F.pad(y, (2,1), mode='replicate')
        y = F.pad(y, (3,4), mode='constant', value=1.3)
        y = F.pad(y, (1,1))

        z = F.pad(z, (3,4,3,4), mode='reflect')
        z = F.pad(z, (2,1,2,0), mode='replicate')
        z = F.pad(z, (1,0,2,0), mode='constant', value=1.3)
        z = F.pad(z, (3,3,3,3))

        #w = F.pad(w, (1,2,3,4,5,6), mode='reflect')
        w = F.pad(w, (5,0,1,2,0,2), mode='replicate')
        w = F.pad(w, (0,2,2,1,3,4), mode='constant', value=1.3)
        w = F.pad(w, (2,2,2,2,2,2))

        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a0, a1, a2, a3 = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_F_pad.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_pad.pt inputshape=[1,16],[12,2,16],[1,3,12,16],[1,5,7,9,11]")

    # pnnx inference
    import test_F_pad_pnnx
    b0, b1, b2, b3 = test_F_pad_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
