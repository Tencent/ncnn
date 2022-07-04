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

    def forward(self, x, y, z):
        x0, x1, x2 = torch.tensor_split(x, (12, 13))
        y0, y1, y2 = torch.tensor_split(y, 3, dim=1)
        z0, z1 = torch.tensor_split(z, (3,), dim=0)
        return x0, x1, x2, y0, y1, y2, z0, z1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(100)
    y = torch.rand(3, 15)
    z = torch.rand(5, 9, 3)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_tensor_split.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_tensor_split.pt inputshape=[100],[3,15],[5,9,3]")

    # ncnn inference
    import test_torch_tensor_split_ncnn
    b = test_torch_tensor_split_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            print(a0.shape)
            print(b0.shape)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
