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

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = F.pixel_shuffle(x, 4)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 6, 7)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_pixel_shuffle.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_pixel_shuffle.pt inputshape=[1,128,6,7]")

    # ncnn inference
    import test_F_pixel_shuffle_ncnn
    b = test_F_pixel_shuffle_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
