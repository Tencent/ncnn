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

    def forward(self, x0, x1, y0, y1, z0, z1, w0, w1, s0, s1, t0, t1):
        x = torch.matmul(x0, x1)
        y = torch.matmul(y0, y1)
        z = torch.matmul(z0, z1)
        w = torch.matmul(w0, w1)
        s = torch.matmul(s0, s1)
        t = torch.matmul(t0, t1)
        return x, y, z, w, s, t

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(13)
    x1 = torch.rand(13)
    y0 = torch.rand(5, 12)
    y1 = torch.rand(12)
    z0 = torch.rand(10, 3, 4)
    z1 = torch.rand(4)
    w0 = torch.rand(10, 23, 14)
    w1 = torch.rand(10, 14, 5)
    s0 = torch.rand(10, 23, 14)
    s1 = torch.rand(14, 5)
    t0 = torch.rand(6, 9, 13, 14)
    t1 = torch.rand(6, 9, 14, 15)

    a = net(x0, x1, y0, y1, z0, z1, w0, w1, s0, s1, t0, t1)

    # export torchscript
    mod = torch.jit.trace(net, (x0, x1, y0, y1, z0, z1, w0, w1, s0, s1, t0, t1))
    mod.save("test_torch_mean.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_mean.pt inputshape=[13],[13],[5,12],[12],[10,3,4],[4],[10,23,14],[10,14,5],[10,23,14],[14,5],[6,9,13,14],[6,9,14,15]")

    # pnnx inference
    import test_torch_mean_pnnx
    b = test_torch_mean_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
