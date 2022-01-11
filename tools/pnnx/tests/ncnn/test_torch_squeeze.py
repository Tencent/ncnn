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
        x = torch.squeeze(x, 0)
        y = torch.squeeze(y, 1)
        z = torch.squeeze(z)
        w = torch.squeeze(w, 2)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(3, 1)
    z = torch.rand(5, 1, 11)
    w = torch.rand(5, 9, 1)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_squeeze.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_squeeze.pt inputshape=[1,16],[3,1],[5,1,11],[5,9,1]")

    # ncnn inference
    import test_torch_squeeze_ncnn
    b = test_torch_squeeze_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            print(a0.shape)
            print(b0.shape)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
